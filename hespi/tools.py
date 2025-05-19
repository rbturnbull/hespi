import typer
from rich.console import Console
from pathlib import Path

from .ocr import TrOCRSize
from .hespi import DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS, DEFAULT_SHEET_COMPONENT_WEIGHTS, DEFAULT_LABEL_FIELD_WEIGHTS
from .download import get_location
from .util import DATA_DIR, ocr_data_json

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
def institutional_label_classifier_location():
    """ Shows the location of the default Institutional Label Classifier model weights. """
    path = get_location(DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS)
    console.print(f"The location of the default Institutional Label Classifier model is:\n{path}")


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


@app.command()
def gui():
    """
    Start the Hespi GUI
    """
    from .gui import build_blocks, compile_sass
    compile_sass(Path(__file__).parent / "templates" / "assets")
    interface = build_blocks()
    interface.launch()
    

@app.command()
def picklejson():
    """
    Converts a pickled HESPI output to a JSON file.
    """
    import pickle
    import json
    from pathlib import Path

    out_dir = Path().cwd() / "hespi-output"
    input_file = out_dir/"ocr_data.pkl"
    output_file = out_dir/"ocr_data_pickle.json"

    with open(str(input_file), "rb") as f:
        ocr_data = pickle.load(f)
        print(ocr_data)
    ocr_data_json(ocr_data, output_path = output_file)

    console.print(f"Converted {input_file} to {output_file}")
