from typing import List 
import typer
from pathlib import Path
from yolov5 import YOLOv5
import pytesseract
from torchapp.examples.image_classifier import ImageClassifier
from rich.console import Console
from .yolo import yolo_output


console = Console()

app = typer.Typer()


@app.command()
def detect(
    images:List[Path] = typer.Argument(..., help="A list of images to process."), 
    output_dir:Path = typer.Argument(..., help="A directory to output the results."),
    gpu:bool = typer.Option(True, help="Whether or not to use a GPU if available."),
):
    """
    HErbarium Specimen sheet PIpeline

    Takes a herbarium specimen sheet image and:
    - detects components such as the institutional label, swatch, etc.
    - classifies whether the institutional label is printed, typed, handwritten or a combination.
    - detects the fields of the institutional label
    - attempts OCR/HTR on the institutional label fields

    Args:
        images (List[Path]): A list of images to process.
        output_dir (Path): A directory to output the results.
    """
    console.print(f"Processing {images}")

    # TODO check if gpu is available
    device = "cpu"

    # Sheet-Components Detection Model
    sheet_component_weights_path = "sheet-component-weights.pt"
    sheet_component_model = YOLOv5(sheet_component_weights_path, device)

    # Institutional-Label-Fields Detection Model
    institutional_label_fields_weights_path = "institutional-label-fields.pt"
    institutional_label_fields_model = YOLOv5(institutional_label_fields_weights_path, device)

    # ImageClassifier
    institutional_label_classifier = ImageClassifier()
    institutional_label_classifier_weights = "institutional-label-classifier.pkl"

    # Sheet-Components predictions
    component_files = yolo_output( sheet_component_model, images, output_dir=output_dir )

    # Institutional Label Field Detection Model Predictions
    for stub, components in component_files.items():
        for component in components:
            if component.name.endswith("institutional_label.jpg"):
                field_files = yolo_output( institutional_label_fields_model, [component], output_dir=output_dir/stub )

                # Institutional Label Classification
                # classifier_results = institutional_label_classifier([component], pretrained=institutional_label_classifier_weights)
                # breakpoint()

                # Tesseract OCR
                for institution_stub, fields in field_files.items():
                    for field in fields:
                        print("field_file:", field)
                        text = pytesseract.image_to_string(str(field)).strip()
                        if text:
                            text_path = field.parent/(field.name[:-3] + "txt")
                            print(f"Writing '{text}' to '{text_path}'")
                            text_path.write_text(text+"\n")
                
                # TODO HTR

    print('csv output')
    # Report