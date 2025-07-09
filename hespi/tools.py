import typer
from rich.console import Console
from pathlib import Path

from .ocr import TrOCRSize
from .hespi import DEFAULT_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS, DEFAULT_SHEET_COMPONENT_WEIGHTS, DEFAULT_LABEL_FIELD_WEIGHTS
from .download import get_location
from .util import DATA_DIR

console = Console()

app = typer.Typer()


@app.command()
def sheet_component_location():
    """ Shows the location of the default Sheet-Component model. """
    path = get_location(DEFAULT_SHEET_COMPONENT_WEIGHTS)
    console.print(f"The location of the default Sheet-Component model is:\n{path}")


@app.command()
def label_field_location():
    """ Shows the location of the default Label-Field model weights. """
    path = get_location(DEFAULT_LABEL_FIELD_WEIGHTS)
    console.print(f"The location of the default Label-Field model is:\n{path}")    


@app.command()
def primary_specimen_label_classifier_location():
    """ Shows the location of the default Primary Specimen Label Classifier model weights. """
    path = get_location(DEFAULT_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS)
    console.print(f"The location of the default Primary Specimen Label Classifier model is:\n{path}")


@app.command()
def bibtex():
    """ Shows the references in BibTeX format. """
    references = DATA_DIR / "references.bib"
    console.print(references.read_text())


@app.command()
def trocr(
    image:Path,
    size: TrOCRSize = typer.Option(
        TrOCRSize.LARGE.value,
        help="The size of the TrOCR model to use for handwritten text recognition.",
        case_sensitive=False,
    ),
):
    """ Run the TrOCR model on an image and print the recognized text. """
    from .ocr import TrOCR

    ocr = TrOCR(size=size)
    text = ocr.get_text(image)
    console.print(text)