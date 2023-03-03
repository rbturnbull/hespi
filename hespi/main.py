from typing import List
import typer
from pathlib import Path
import pandas as pd
from rich.console import Console

from .hespi import Hespi
from .hespi import DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS, DEFAULT_SHEET_COMPONENT_WEIGHTS, DEFAULT_INSTITUTIONAL_LABEL_FIELDS_WEIGHTS
from .ocr import TrOCRSize

console = Console()

app = typer.Typer()

@app.command()
def detect(
    images: List[str] = typer.Argument(..., help="A list of images to process."),
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
        DEFAULT_SHEET_COMPONENT_WEIGHTS, 
        help="The path to the sheet component model weights.",
    ),
    institutional_label_fields_weights: str = typer.Option(
        DEFAULT_INSTITUTIONAL_LABEL_FIELDS_WEIGHTS,
        help="The path to the institutional label field model weights.",
    ),
    institutional_label_classifier_weights: str = typer.Option(
        DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS,
        help="The path to the institutional label classifier weights.",
    ),
    force_download:bool = typer.Option(False, help="Whether or not to force download model weights even if a weights file is present."),
    tmp_dir:str = None,
    batch_size:int = typer.Option(4, min=1, help="The maximum batch size from run the sheet component model."),
) -> pd.DataFrame:
    """
    HErbarium Specimen sheet PIpeline

    Takes a herbarium specimen sheet image detects components such as the institutional label, swatch, etc.
    It then classifies whether the institutional label is printed, typed, handwritten or a combination.
    If then detects the fields of the institutional label and attempts to read them through OCR and HTR models.
    """
    hespi = Hespi(
        trocr_size=trocr_size,
        sheet_component_weights=sheet_component_weights,
        institutional_label_fields_weights=institutional_label_fields_weights,
        institutional_label_classifier_weights=institutional_label_classifier_weights,
        force_download=force_download,
        gpu=gpu,
        fuzzy=fuzzy,
        fuzzy_cutoff=fuzzy_cutoff,
        htr=htr,
        tmp_dir=tmp_dir,
        batch_size=batch_size,
    )
    return hespi.detect(images, output_dir)
    