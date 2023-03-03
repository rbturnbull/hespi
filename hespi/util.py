from pathlib import Path
from typing import Dict
import pandas as pd
from rich.console import Console
from difflib import get_close_matches

console = Console()


institutional_label_fields = [
    "family",
    "genus",
    "species",
    "infrasp_taxon",
    "authority",
    "collector_number",
    "collector",
    "locality",
    "geolocation",
    "year",
    "month",
    "day",
]

def adjust_case(field, value):
    if field in ["genus", "family"]:
        return value.title()
    elif field == "species":
        return value.lower()
    
    return value


def read_reference(field):
    DATA_DIR = Path(__file__).parent / "data"
    path = DATA_DIR / f"{field}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No reference file for field '{field}'.")
    return path.read_text().strip().split("\n")


def ocr_data_df(data: dict, output_path: Path=None) -> pd.DataFrame:
    """    
    Creates a DataFrame of data, sorts columns and outputs a CSV with OCR values.

    Args:
        data (dict): The data coming from the text recognition models. 
            The keys are the institutional labels.
            The values are dictionaries with the fields as keys and the values as the recognised text.
        output_path (Path, optional): Where to save a CSV file if given. Defaults to None.

    Returns:
        pd.DataFrame: The text recognition data as a Pandas dataframe
    """
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reset_index().rename(columns={"index": "institutional label"})
    
    # insert columns not included in dataframe, and re-order
    # including any columns not included in col_options to account for any updates
    col_options = [ "institutional label", "id" ] + institutional_label_fields

    missing_cols = [col for col in col_options if col not in df.columns]
    df[missing_cols] = ""
    extra_cols = [col for col in df.columns if col not in col_options]
    cols = col_options + extra_cols
    df = df[cols]
    df = df.fillna('')
    
    # CSV output
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)

    return df


def adjust_text(field:str, recognised_text:str, fuzzy:bool, fuzzy_cutoff:float, reference:Dict):
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
    return text_adjusted


def get_stub(path:Path) -> str:
    """
    Gets the part of the filename before the last extension

    Args:
        path (Path): The path to the file

    Returns:
        str: the part of the filename before the last extension
    """
    last_period = path.name.rfind(".")
    stub = path.name[:last_period] if last_period else path.name
    return stub

    