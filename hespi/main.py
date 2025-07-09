from typing import List
import typer
from pathlib import Path
import pandas as pd
from rich.console import Console

from .hespi import Hespi
from .hespi import DEFAULT_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS, DEFAULT_SHEET_COMPONENT_WEIGHTS, DEFAULT_LABEL_FIELD_WEIGHTS
from .ocr import TrOCRSize

console = Console()

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def detect(
    images: List[str] = typer.Argument(..., help="A list of images to process. The images can also be URLs."),
    output_dir: Path = typer.Option(
        "hespi-output",
        help="A directory to output the results.",
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
    llm: str = typer.Option(
        "gpt-4o",
        help="The Large Langauge Model to use. Currently OpenAI and Anthropic Claude models supported.",
    ),
    llm_api_key: str = typer.Option(
        "",
        help="The API key to use for the Large Language Model. Can be set as an environment variable using the standard variable names.",
    ),
    llm_temperature: float = typer.Option(
        0.0,
        help="The temperature to use for the Large Language Model.",
    ),
    trocr_size: TrOCRSize = typer.Option(
        TrOCRSize.LARGE.value,
        help="The size of the TrOCR model to use for handwritten text recognition.",
        case_sensitive=False,
    ),
    sheet_component_weights: str = typer.Option(
        DEFAULT_SHEET_COMPONENT_WEIGHTS, 
        help="The path to the sheet component model weights.",
        envvar="HESPI_SHEET_COMPONENT_WEIGHTS",
    ),
    label_field_weights: str = typer.Option(
        DEFAULT_LABEL_FIELD_WEIGHTS,
        help="The path to the Label-Field model weights.",
        envvar="HESPI_LABEL_FIELD_WEIGHTS",
    ),
    primary_specimen_label_classifier_weights: str = typer.Option(
        DEFAULT_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS,
        envvar="HESPI_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS",
        help="The path to the primary specimen label classifier weights.",
    ),
    force_download:bool = typer.Option(False, help="Whether or not to force download model weights even if a weights file is present."),
    batch_size:int = typer.Option(4, min=1, help="The maximum batch size from run the sheet component model."),
    sheet_component_res:int = typer.Option(1280, min=640, help="The resolution of images to use for the Sheet-Component model."),
    label_field_res:int = typer.Option(1280, min=640, help="The resolution of images to use for the Label-Field model."),
) -> pd.DataFrame:
    """
    HErbarium Specimen sheet PIpeline

    Takes a herbarium specimen sheet image detects components such as the primary specimen label, swatch, etc.
    It then classifies whether the primary specimen label is printed, typed, handwritten or a combination.
    If then detects the fields of the primary specimen label and attempts to read them through OCR and HTR models.
    """
    hespi = Hespi(
        trocr_size=trocr_size,
        sheet_component_weights=sheet_component_weights,
        label_field_weights=label_field_weights,
        primary_specimen_label_classifier_weights=primary_specimen_label_classifier_weights,
        force_download=force_download,
        gpu=gpu,
        fuzzy=fuzzy,
        fuzzy_cutoff=fuzzy_cutoff,
        llm_model=llm,
        llm_api_key=llm_api_key,
        llm_temperature=llm_temperature,
        htr=htr,
        batch_size=batch_size,
        sheet_component_res=sheet_component_res,
        label_field_res=label_field_res,
    )
    return hespi.detect(images, output_dir)
    