import typer
from rich.console import Console

from .hespi import DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS, DEFAULT_SHEET_COMPONENT_WEIGHTS, DEFAULT_LABEL_FIELD_WEIGHTS
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
def bibtex():
    """ Shows the references in BibTeX format. """
    references = DATA_DIR / "references.bib"
    console.print(references.read_text())
