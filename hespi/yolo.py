import tempfile
from shutil import move
from pathlib import Path
from PIL import Image
from collections import defaultdict
from rich.console import Console


try:
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
except ImportError: # pragma: no cover
    from ultralytics.yolo.v8.detect.predict import DetectionPredictor # pragma: no cover


console = Console()

from .util import get_stub

def predictions_filename(stub):
    return f"{stub}-predictions.jpg"


def yolo_output(model, images, output_dir, tmp_dir_prefix=None, batch_size=4, res=1280):
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

    for index, (image, predictions) in enumerate(zip(images, results)):
        stub = get_stub(image)
        image_output_dir = output_dir / stub
        image_output_dir.mkdir(exist_ok=True, parents=True)
        prediction_path = image_output_dir / predictions_filename(stub)

        prediction_path_tmp_location = Path(model.predictor.save_dir)/image.name
        assert prediction_path_tmp_location.exists()

        console.print(
            f"Saving sheet component predicitons with bounding boxes to '{prediction_path}'"
        )
        move(prediction_path_tmp_location, prediction_path)

        for prediction_index, boxes in enumerate(predictions.boxes):
            category_index = int(boxes.cls.cpu().item())
            category = (
                model.names[category_index].replace(" ", "_").replace(":", "").strip()
            )

            assert len(boxes.xyxy) == 1
            x0, y0, x1, y1 = boxes.xyxy.cpu().numpy()[0]

            # open image
            im = Image.open(image)
            im_crop = im.crop((x0, y0, x1, y1))
            output_path = (
                image_output_dir / f"{stub}.{prediction_index}.{category}.jpg"
            )
            console.print(f"Saving {category} to '{output_path}'")
            im_crop.convert('RGB').save(output_path)
            output_files[stub].append(output_path)

    tmp_dir.cleanup()

    return output_files