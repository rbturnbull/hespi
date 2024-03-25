import tempfile
from shutil import move
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Column, Table

console = Console()

from .util import get_stub

def predictions_filename(stub):
    return f"{stub}.all.jpg"


def yolo_output(model, images, output_dir, tmp_dir_prefix=None, batch_size=4, res=1280):
    try:
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
    except ImportError: # pragma: no cover
        from ultralytics.yolo.v8.detect.predict import DetectionPredictor # pragma: no cover

    if not model.predictor:
        model.predictor = DetectionPredictor()
        model.predictor.setup_model(model=model.model)

    tmp_dir = tempfile.TemporaryDirectory(prefix=tmp_dir_prefix)
    tmp_dir_path = Path(tmp_dir.name)
    console.print(f"Using temporary directory '{tmp_dir_path}'")

    model.predictor.save_dir = tmp_dir_path

    results = model.predict(source=images, show=False, save=True, batch=batch_size, imgsz=res)
    output_files = defaultdict(list)

    output_dir = Path(output_dir)

    for image, predictions in zip(images, results):
        stub = get_stub(image)
        image_output_dir = output_dir / stub
        image_output_dir.mkdir(exist_ok=True, parents=True)
        prediction_path = image_output_dir / predictions_filename(stub)

        prediction_path_tmp_location = Path(model.predictor.save_dir)/image.name
        assert prediction_path_tmp_location.exists()

        table = Table(
            Column(header="Category", justify="middle"),
            Column(header=f"File in directory '{image_output_dir}'", justify="left", style="green"),
            title=f"Saving predicitons for: '{stub}'",
        )
        table.add_row("All", prediction_path.name)

        move(prediction_path_tmp_location, prediction_path)

        counter = Counter()

        for _, boxes in enumerate(predictions.boxes):
            category_index = int(boxes.cls.cpu().item())
            category = (
                model.names[category_index].replace(" ", "_").replace(":", "").strip()
            )

            assert len(boxes.xyxy) == 1
            x0, y0, x1, y1 = boxes.xyxy.cpu().numpy()[0]

            # open image
            im = Image.open(image)
            im_crop = im.crop((x0, y0, x1, y1))
            counter.update([category])

            counter_suffix = f"-{counter[category]}" if counter[category] > 1 else ""
            
            output_path = (
                image_output_dir / f"{stub}.{category}{counter_suffix}.jpg"
            )
            table.add_row(category, output_path.name)

            im_crop.convert('RGB').save(output_path)
            output_files[stub].append(output_path)

    tmp_dir.cleanup()

    console.print(table)

    return output_files