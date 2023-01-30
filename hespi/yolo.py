import tempfile
from shutil import move
from pathlib import Path
from PIL import Image
from collections import defaultdict
from rich.console import Console

console = Console()


def yolo_output_batch(model, images, output_dir, tmp_dir_prefix=None):
    results = model.predict(images)
    tmp_dir = tempfile.TemporaryDirectory(prefix=tmp_dir_prefix)
    tmp_dir_path = Path(tmp_dir.name)
    console.print(f"Using temporary directory '{tmp_dir_path}'")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

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
            category_index = int(prediction[5].int().cpu().numpy())
            category = (
                results.names[category_index].replace(" ", "_").replace(":", "").strip()
            )

            x0, y0, x1, y1 = prediction[:4].cpu().numpy()

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


def yolo_output(model, images, output_dir, tmp_dir_prefix=None, batch_size=4):
    output_files = defaultdict(list)
    images_count = len(images)
    # based on https://stackoverflow.com/a/8290508
    for start in range(0, images_count, batch_size):
        batch = images[start:min(start+batch_size, images_count)]
        batch_result = yolo_output_batch(model, batch, output_dir=output_dir, tmp_dir_prefix=tmp_dir_prefix)
        output_files.update(batch_result)
    return output_files