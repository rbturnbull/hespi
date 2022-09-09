from typing import List
import typer
from pathlib import Path
import pandas as pd
from difflib import get_close_matches
import torch
from yolov5 import YOLOv5
from torchapp.examples.image_classifier import ImageClassifier
from rich.console import Console

from .yolo import yolo_output
from .ocr import Tesseract, TrOCR, TrOCRSize
from .download import get_weights

console = Console()

app = typer.Typer()


def adjust_case(field, value):
    if field in ["genus", "family"]:
        return value.title()
    if field == "species":
        return value.lower()
    return value


def read_reference(field):
    DATA_DIR = Path(__file__).parent / "data"
    path = DATA_DIR / f"{field}.txt"
    return path.read_text().split("\n")


@app.command()
def detect(
    images: List[Path] = typer.Argument(..., help="A list of images to process."),
    output_dir: Path = typer.Option(
        ...,
        help="A directory to output the results.",
        prompt="Please specify a directory for the results",
    ),
    gpu: bool = typer.Option(True, help="Whether or not to use a GPU if available."),
    fuzzy: bool = typer.Option(
        True, help="Whether or not to use fuzzy matching from teh reference database."
    ),
    fuzzy_cutoff: float = typer.Option(
        0.8, min=0.0, max=1.0, help="The threshold for the fuzzy matching score to use."
    ),
    htr: bool = typer.Option(
        True,
        help="Whether or not to do handwritten text recognition using Microsoft's TrOCR.",
    ),
    trocr_size: TrOCRSize = typer.Option(
        TrOCRSize.BASE.value,
        help="The size of the TrOCR model to use for handwritten text recognition.",
        case_sensitive=False,
    ),
    sheet_component_weights: str = typer.Option(
        "https://github.com/rbturnbull/hespi/releases/download/v0.1.0-alpha/sheet-component-weights.pt",
        help="The path to the sheet component model weights.",
    ),
    institutional_label_fields_weights: str = typer.Option(
        "https://github.com/rbturnbull/hespi/releases/download/v0.1.0-alpha/institutional-label-fields.pt",
        help="The path to the institutional label field model weights.",
    ),
    institutional_label_classifier_weights: str = typer.Option(
        "https://github.com/rbturnbull/hespi/releases/download/v0.1.0-alpha/institutional-label-classifier.pkl",
        help="The path to the institutional label classifier weights.",
    ),
    force_download:bool = typer.Option(False, help="Whether or not to force download model weights even if a weights file is present."),
):
    """
    HErbarium Specimen sheet PIpeline

    Takes a herbarium specimen sheet image detects components such as the institutional label, swatch, etc.
    It then classifies whether the institutional label is printed, typed, handwritten or a combination.
    If then detects the fields of the institutional label and attempts to read them through OCR and HTR models.

    Args:
        images (List[Path]): A list of images to process.
        output_dir (Path): A directory to output the results.
        gpu (bool): Whether or not to use a GPU if available. Default True.
    """
    console.print(f"Processing {images}")

    # Check if gpu is available
    gpu = gpu and torch.cuda.is_available()
    device = "cuda:0" if gpu else "cpu"

    # Sheet-Components Detection Model
    sheet_component_weights = get_weights(sheet_component_weights, force=force_download)
    sheet_component_model = YOLOv5(sheet_component_weights, device)

    # Institutional-Label-Fields Detection Model
    institutional_label_fields_weights = get_weights(institutional_label_fields_weights, force=force_download)
    institutional_label_fields_model = YOLOv5(
        institutional_label_fields_weights, device
    )

    # ImageClassifier
    institutional_label_classifier_weights = get_weights(institutional_label_classifier_weights, force=force_download)
    institutional_label_classifier = ImageClassifier()

    reference_fields = ["family", "genus", "species", "authority"]
    reference = {field: read_reference(field) for field in reference_fields}

    # OCR models
    tesseract = Tesseract()
    if htr:
        print(f"Getting TrOCR model of size '{trocr_size}'")
        trocr = TrOCR(size=trocr_size)
        print("TrOCR model available")

    # Sheet-Components predictions
    component_files = yolo_output(sheet_component_model, images, output_dir=output_dir)

    ocr_data = {}
    # Institutional Label Field Detection Model Predictions
    for stub, components in component_files.items():
        for component in components:
            if component.name.endswith("institutional_label.jpg"):
                field_files = yolo_output(
                    institutional_label_fields_model,
                    [component],
                    output_dir=output_dir / stub,
                )
                row = {
                    "id": stub,
                }

                # Institutional Label Classification
                classification_csv = (
                    component.parent / f"{component.name[:3]}.classification.csv"
                )
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
                if isinstance(classifier_results, pd.DataFrame):
                    classification = classifier_results.iloc[0]["prediction"]
                    row["label_classification"] = classification
                    console.print(
                        f"'{component}' classified as '[red]{classification}[/red]'."
                    )
                else:
                    console.print(
                        f"Could not get classification of institutional label '{component}'"
                    )
                    classification = None

                # Text Recognition on bounding boxes found by Yolo
                for fields in field_files.values():
                    for field_file in fields:
                        console.print("field_file:", field_file)
                        field_file_components = field_file.name.split(".")
                        assert len(field_file_components) >= 5
                        field = field_file_components[-2]

                        # HTR
                        recognised_text = ""
                        if htr:
                            htr_text = trocr.get_text(field_file)
                            if htr_text:
                                print(f"HTR: {htr_text}")
                                console.print(
                                    f"Handwritten Text Recognition (TrOCR): [red]'{htr_text}'[/red]"
                                )

                                row[f"{field}_TrOCR"] = htr_text
                                recognised_text = htr_text

                        # OCR
                        tesseract_text = tesseract.get_text(field_file)
                        if tesseract_text:
                            row[f"{field}_tesseract"] = tesseract_text
                            console.print(
                                f"Optical Character Recognition (Tesseract): [red]'{tesseract_text}'[/red]"
                            )

                            if (
                                classification
                                and classification in ["printed", "typewriter"]
                            ) or not recognised_text:
                                recognised_text = tesseract_text

                        # Adjust text if necessary
                        if recognised_text:
                            # Adjust case
                            text_adjusted = adjust_case(field, recognised_text)

                            # Match with database
                            if fuzzy and field in reference:
                                close_matches = get_close_matches(
                                    text_adjusted,
                                    reference[field],
                                    cutoff=fuzzy_cutoff,
                                    n=1,
                                )
                                if close_matches:
                                    text_adjusted = close_matches[0]

                            if recognised_text != text_adjusted:
                                console.print(
                                    f"Recognized text [red]'{recognised_text}'[/red] adjusted to [purple]'{text_adjusted}'[/purple]"
                                )

                            row[field] = text_adjusted

                ocr_data[str(component)] = row

    csv_creation(ocr_data, output_dir)


def csv_creation(data: dict, output_dir: Path):
    """
    Creates a DataFrame of data, sorts columns and outputs a CSV with OCR values.
    """
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reset_index().rename(columns={"index": "institutional label"})

    # insert columns not included in dataframe, and re-order
    # including any columns not included in col_options to account for any updates
    col_options = [
        "institutional label",
        "id",
        "family",
        "genus",
        "species",
        "infrasp taxon",
        "authority",
        "collector_number",
        "collector",
        "locality",
        "geolocation",
        "year",
        "month",
        "day",
    ]

    missing_cols = [col for col in col_options if col not in df.columns]
    df[missing_cols] = ""
    extra_cols = [col for col in df.columns if col not in col_options]
    cols = col_options + extra_cols
    df = df[cols]

    # CSV output
    df.to_csv(str(output_dir) + "/ocr_results.csv", index=False)
    return df
