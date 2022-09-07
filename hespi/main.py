from typing import List 
import typer
from pathlib import Path
from yolov5 import YOLOv5
from difflib import get_close_matches
import torch
import pytesseract
from torchapp.examples.image_classifier import ImageClassifier
from rich.console import Console
import pandas as pd
from .yolo import yolo_output
import pandas as pd
import numpy as np


console = Console()

app = typer.Typer()

def adjust_case(field, value):
    if field in ['genus', 'family']:
        return value.title()
    if field == "species":
        return value.lower()
    return value


def read_reference(field):
    DATA_DIR = Path(__file__).parent/"data"
    path = DATA_DIR/f"{field}.txt"
    return path.read_text().split("\n")


@app.command()
def detect(
    images:List[Path] = typer.Argument(..., help="A list of images to process."), 
    output_dir:Path = typer.Option(..., help="A directory to output the results.", prompt="Please specify a directory for the results"),
    gpu:bool = typer.Option(True, help="Whether or not to use a GPU if available."),
    fuzzy:bool = typer.Option(True, help="Whether or not to use fuzzy matching from teh reference database."),
    fuzzy_cutoff:float = typer.Option(0.8, min=0.0, max=1.0, help="The threshold for the fuzzy matching score to use."),
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
        gpu (bool): Whether or not to use a GPU if available. Default True.
    """
    console.print(f"Processing {images}")

    # Check if gpu is available
    gpu = gpu and torch.cuda.is_available()
    device = "gpu" if gpu else "cpu"

    # Sheet-Components Detection Model
    sheet_component_weights_path = "sheet-component-weights.pt"
    sheet_component_model = YOLOv5(sheet_component_weights_path, device)

    # Institutional-Label-Fields Detection Model
    institutional_label_fields_weights_path = "institutional-label-fields.pt"
    institutional_label_fields_model = YOLOv5(institutional_label_fields_weights_path, device)

    # ImageClassifier
    institutional_label_classifier = ImageClassifier()
    institutional_label_classifier_weights = "institutional-label-classifier.pkl"

    reference_fields = ['family', 'genus', 'species', 'authority']
    reference = {field:read_reference(field) for field in reference_fields}

    # Sheet-Components predictions
    component_files = yolo_output( sheet_component_model, images, output_dir=output_dir )

    ocr_data = {}
    # Institutional Label Field Detection Model Predictions
    for stub, components in component_files.items():
        for component in components:
            if component.name.endswith("institutional_label.jpg"):
                field_files = yolo_output( institutional_label_fields_model, [component], output_dir=output_dir/stub )
                row = {
                    "id":stub,
                }

                # Institutional Label Classification
                classification_csv = component.parent/f"{component.name[:3]}.classification.csv"
                console.print(f"Classifying institution label: '{component}'")
                console.print(f"Saving result to '{classification_csv}'")
                classifier_results = institutional_label_classifier(
                    items=component, 
                    pretrained=institutional_label_classifier_weights, 
                    gpu=gpu, 
                    output_csv=classification_csv,
                    verbose=False,
                )

                # Get classification of institution label
                # 
                if isinstance(classifier_results, pd.DataFrame):
                    classiciation = classifier_results.iloc[0]['prediction']
                    row['label_classification'] = classiciation
                    console.print(f"'{component}' classified as '[red]{classiciation}[/red]'.")
                else:
                    console.print(f"Could not get classification of institutional label '{component}'")
                    classiciation = None

                # Tesseract OCR
                for institution_stub, fields in field_files.items():
                    for field_file in fields:
                        console.print("field_file:", field_file)
                        ocr_text = pytesseract.image_to_string(str(field_file)).strip()
                        if ocr_text:
                            field_file_components = field_file.name.split(".")
                            assert len(field_file_components) >= 5
                            field = field_file_components[-2]

                            text_path = field_file.parent/(field_file.name[:-3] + "txt")
                            console.print(f"Writing [red]'{ocr_text}'[/red] to '{text_path}'")
                            text_path.write_text(ocr_text+"\n")

                            # Adjust case
                            text_adjusted = adjust_case(field, ocr_text)

                            # Match with database
                            if fuzzy and field in reference:
                                close_matches = get_close_matches(text_adjusted, reference[field], cutoff=fuzzy_cutoff, n=1)
                                if close_matches:
                                    text_adjusted = close_matches[0]

                            if ocr_text != text_adjusted:
                                console.print(f"OCR text [red]'{ocr_text}'[/red] adjusted to [purple]'{text_adjusted}'[/purple]")
                                breakpoint()

                            row[field] = text_adjusted
                            row[f"{field}_ocr"] = ocr_text

                # TODO HTR
                
                ocr_data[str(component)] = row

    csv_creation(ocr_data, output_dir)



def csv_creation(data:dict, output_dir:Path):
    """
    Creates a DataFrame of data, checks if OCR output is a known value, and outputs a CSV with OCR values
    """

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index().rename(columns={"index": "institutional label"})

    # insert columns not included in dataframe, and re-order
    # including any columns not included in col_options to account for any updates
    col_options = ['institutional label', "id", 'family', 'genus', 'species', 'infrasp taxon', 
                    'authority', 'collector_number', 'collector', 
                    'locality', 'geolocation', 'year', 'month', 'day']

    missing_cols = [col for col in col_options if col not in df.columns]
    df[missing_cols] = ''
    extra_cols = [col for col in df.columns if col not in col_options]
    cols = col_options + extra_cols
    df = df[cols]

    # CSV output
    df.to_csv(str(output_dir)+'/ocr_results.csv', index=False) 
    return df