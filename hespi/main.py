from typing import List 
import typer
from pathlib import Path
from rich.console import Console
from yolov5 import YOLOv5
import tempfile
from shutil import move
from PIL import Image
from collections import defaultdict

console = Console()

app = typer.Typer()

def yolo_output(model, images, output_dir):
    results = model.predict(images)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)
    # TODO check that all images have unique names
    
    results.save( save_dir=tmp_dir_path )

    output_files = defaultdict(list)

    for image, predictions in zip(images, results.pred):
        last_period = image.name.rfind(".")
        stub = image.name[:last_period] if last_period else image.name
        image_output_dir = output_dir/stub
        image_output_dir.mkdir(exist_ok=True, parents=True)
        prediction_path = image_output_dir/f"{stub}-predictions.jpg"
        console.print(f"Saving sheet component predicitons with bounding boxes to '{prediction_path}'")
        move(tmp_dir_path/f"{stub}.jpg", prediction_path)

        for prediction_index, prediction in enumerate(predictions):
            category_index = prediction[5].int().numpy()
            category = results.names[category_index].replace(" ", "_").replace(":","").strip()

            x0, y0, x1, y1 = prediction[:4].numpy()
            
            # open image
            im = Image.open(image)
            im_crop = im.crop((x0, y0, x1, y1))
            output_filename = image_output_dir/f"{stub}.{prediction_index}.{category}.jpg"
            console.print(f"Saving {category} to '{output_filename}'")
            im_crop.save(output_filename)    
            output_files[stub].append(output_filename)
    
    tmp_dir.cleanup()

    return output_files


@app.command()
def main(images:List[Path], output_dir:Path=Path("output")):
    """
    HErbarium SPecimen sheet Inference

    Takes a herbarium specimen sheet image and:
    - detects components such as the institutional label, swatch, etc.
    - classifies whether the institutional label is printed, typed, handwritten or a combination.
    - detects the fields of the institutional label
    - attempts OCR/HTR on the institutional label fields
    """
    console.print(f"Processing {images}")

    device = "cpu"

    # Sheet-Components Detection Model
    sheet_component_weights_path = "sheet-component-weights.pt"
    sheet_component_model = YOLOv5(sheet_component_weights_path, device)

    # Institutional-Label-Fields Detection Model
    institutional_label_fields_weights_path = "institutional-label-fields.pt"
    institutional_label_fields_model = YOLOv5(institutional_label_fields_weights_path, device)

    # Sheet-Components predictions
    component_files = yolo_output( sheet_component_model, images, output_dir=output_dir )

    # Institutional Label Field Detection Model Predictions
    for stub, components in component_files.items():
        for component in components:
            if component.name.endswith("institutional_label.jpg"):
                field_files = yolo_output( institutional_label_fields_model, [component], output_dir=output_dir/stub/ )

                # TODO Institutional Label Classification
