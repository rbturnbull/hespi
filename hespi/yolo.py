import tempfile
from shutil import move
from pathlib import Path
from PIL import Image
from collections import defaultdict
from rich.console import Console

console = Console()


def yolo_output(model, images, output_dir):
    results = model.predict(images)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)
    # TODO check that all images have unique names
    # TODO download from internet if it is given a URL

    results.save(save_dir=tmp_dir_path)

    output_files = defaultdict(list)

    for image, predictions in zip(images, results.pred):
        last_period = image.name.rfind(".")
        stub = image.name[:last_period] if last_period else image.name
        image_output_dir = output_dir / stub
        image_output_dir.mkdir(exist_ok=True, parents=True)
        prediction_path = image_output_dir / f"{stub}-predictions.jpg"
        console.print(
            f"Saving sheet component predicitons with bounding boxes to '{prediction_path}'"
        )
        move(tmp_dir_path / f"{stub}.jpg", prediction_path)

        for prediction_index, prediction in enumerate(predictions):
            category_index = prediction[5].int().cpu().numpy()
            category = (
                results.names[category_index].replace(" ", "_").replace(":", "").strip()
            )

            x0, y0, x1, y1 = prediction[:4].cpu().numpy()

            # open image
            im = Image.open(image)
            im_crop = im.crop((x0, y0, x1, y1))
            output_filename = (
                image_output_dir / f"{stub}.{prediction_index}.{category}.jpg"
            )
            console.print(f"Saving {category} to '{output_filename}'")
            im_crop.save(output_filename)
            output_files[stub].append(output_filename)

    tmp_dir.cleanup()

    return output_files
