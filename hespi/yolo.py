from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Column, Table
from drawyolo.draw import draw_box_on_image_with_yolo_result


console = Console()

from .util import get_stub

def predictions_filename(stub):
    return f"{stub}.all.jpg"


def yolo_output(model, images, output_dir: str | Path, res: int = 1280, thumbnail_width: int = 240):
    output_files = defaultdict(list)

    output_dir = Path(output_dir)

    for image in images:
        results = model.predict(source=[image], show=False, save=False, batch=1, imgsz=res)
        predictions = next(iter(results))

        stub = get_stub(image)
        image_output_dir = output_dir / stub
        prediction_path = image_output_dir / predictions_filename(stub)
        draw_box_on_image_with_yolo_result(
            image,
            predictions,
            output=prediction_path,
            classes=model.names,
        )
        draw_box_on_image_with_yolo_result(
            image,
            predictions,
            output=image_output_dir/f"{stub}.thumbnail.jpg",
            classes=model.names,
            width=thumbnail_width,
        )
        draw_box_on_image_with_yolo_result(
            image,
            predictions,
            output=image_output_dir/f"{stub}.medium.jpg",
            classes=model.names,
            width=400,
        )

        table = Table(
            Column(header="Category", justify="middle"),
            Column(header=f"File in directory '{image_output_dir}'", justify="left", style="green"),
            title=f"Saving predicitons for: '{stub}'",
        )
        table.add_row("All", prediction_path.name)

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

    console.print(table)

    return output_files
