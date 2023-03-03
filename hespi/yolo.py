import tempfile
from shutil import move
from pathlib import Path
from PIL import Image
from collections import defaultdict
from rich.console import Console

from ultralytics.yolo.v8.detect.predict import DetectionPredictor

console = Console()

from .util import get_stub

def predictions_filename(stub):
    return f"{stub}-predictions.jpg"

def yolov5_output_batch(model, images, output_dir, tmp_dir_prefix=None):
    results = model.predict(images)
    tmp_dir = tempfile.TemporaryDirectory(prefix=tmp_dir_prefix)
    tmp_dir_path = Path(tmp_dir.name)
    console.print(f"Using temporary directory '{tmp_dir_path}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # TODO check that all images have unique names
    # TODO download from internet if it is given a URL
    results.save(save_dir=tmp_dir_path, exist_ok=True)

    output_files = defaultdict(list)

    for image, predictions in zip(images, results.pred):
        stub = get_stub(image)
        image_output_dir = output_dir / stub
        image_output_dir.mkdir(exist_ok=True, parents=True)
        prediction_path = image_output_dir / predictions_filename(stub)
        console.print(
            f"Saving sheet component predicitons with bounding boxes to '{prediction_path}'"
        )
        prediction_path_tmp_location = tmp_dir_path / f"{stub}.jpg"
        move(prediction_path_tmp_location, prediction_path)

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


def yolov5_output(model, images, output_dir, tmp_dir_prefix=None, batch_size=4):
    output_files = defaultdict(list)
    images_count = len(images)
    # based on https://stackoverflow.com/a/8290508
    for start in range(0, images_count, batch_size):
        batch = images[start:min(start+batch_size, images_count)]
        batch_result = yolov5_output_batch(model, batch, output_dir=output_dir, tmp_dir_prefix=tmp_dir_prefix)
        output_files.update(batch_result)
    return output_files


def yolo_output(model, images, output_dir, tmp_dir_prefix=None, batch_size=4):
    if not model.predictor:
        model.predictor = DetectionPredictor()
        model.predictor.setup_model(model=model.model)

    tmp_dir = tempfile.TemporaryDirectory(prefix=tmp_dir_prefix)
    tmp_dir_path = Path(tmp_dir.name)
    console.print(f"Using temporary directory '{tmp_dir_path}'")

    model.predictor.save_dir = tmp_dir_path

    results = model.predict(source=images, show=False, save=True)
    output_files = defaultdict(list)

    output_dir = Path(output_dir)

    for index, (image, predictions) in enumerate(zip(images, results)):
        stub = get_stub(image)
        image_output_dir = output_dir / stub
        image_output_dir.mkdir(exist_ok=True, parents=True)
        prediction_path = image_output_dir / predictions_filename(stub)

        prediction_path_tmp_location = Path(model.predictor.save_dir)/f"image{index}.jpg"
        assert prediction_path_tmp_location.exists()

        console.print(
            f"Saving sheet component predicitons with bounding boxes to '{prediction_path}'"
        )
        move(prediction_path_tmp_location, prediction_path)

        for prediction_index, boxes in enumerate(predictions.boxes.boxes):
            category_index = int(boxes[5].int().cpu().numpy())
            category = (
                model.names[category_index].replace(" ", "_").replace(":", "").strip()
            )

            x0, y0, x1, y1 = boxes[:4].cpu().numpy()

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