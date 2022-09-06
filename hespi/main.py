from typing import List 
import typer
from pathlib import Path
from yolov5 import YOLOv5
import pytesseract
from torchapp.examples.image_classifier import ImageClassifier
from rich.console import Console
import pandas as pd
from .yolo import yolo_output
import pandas as pd
import numpy as np


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

    data = {}
    # Institutional Label Field Detection Model Predictions
    for stub, components in component_files.items():
        for component in components:
            if component.name.endswith("institutional_label.jpg"):
                field_files = yolo_output( institutional_label_fields_model, [component], output_dir=output_dir/stub )
                row = {}

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

                if isinstance(classifier_results, pd.DataFrame):
                    classiciation = classifier_results.iloc[0]['prediction']
                    console.print(f"'{component}' classified as '[red]{classiciation}[/red]'.")
                else:
                    console.print(f"Could not get classification of institutional label '{component}'")
                    classiciation = None

                # Tesseract OCR
                for institution_stub, fields in field_files.items():
                    for field in fields:
                        console.print("field_file:", field)
                        text = pytesseract.image_to_string(str(field)).strip()
                        if text:
                            text_path = field.parent/(field.name[:-3] + "txt")
                            console.print(f"Writing [red]'{text}'[/red] to '{text_path}'")
                            text_path.write_text(text+"\n")

                            row[field.name.split('.')[-2]]=text
                
                data[str(field).split('/')[-1].split('.')[0]] = row
                # TODO HTR

    csv_creation(data, output_dir)

    # Report


def csv_creation(data, output_dir):
    """Function to create a df on data, check if OCR output is a known value,
    and output a csv with OCR values"""

    # create dataframe
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index().rename(columns={"index": "image"})

    # insert columns not included in dataframe, and re-order
    # including any columns not included in col_options to account for any updates
    col_options = ['image', 'family', 'genus', 'species', 'infrasp taxon', 
                    'authority', 'collector_number', 'collector', 
                    'locality', 'geolocation', 'year', 'month', 'day']

    missing_cols = [col for col in col_options if col not in df.columns]
    df[missing_cols] = ''
    extra_cols = [col for col in df.columns if col not in col_options]
    cols = col_options + extra_cols
    df = df[cols]

    # checking to see if species values are known species
    # adding a column with True/False if known
    # removing result if original column is blank for easier reading

    # currently exact matches based of what is in Specify - to be updated

    ref = pd.read_csv('https://raw.githubusercontent.com/rbturnbull/hespi/csv/data/maria_db_plants.csv')
    
    match_cols = ['family', 'genus', 'species', 'authority']

    for col in match_cols:
        df[f"{col}_match"] = np.where(df[col].isin(ref[col]), True, False)
        df.loc[(df[col].isna())|(df[col]==''), f"{col}_match"] = ''

    # CSV output
    df.to_csv(str(output_dir)+'/ocr_results.csv', index=False) 
    return df