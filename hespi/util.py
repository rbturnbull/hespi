from pathlib import Path
from typing import Dict
import pandas as pd
from rich.console import Console
from difflib import get_close_matches, SequenceMatcher

console = Console()

DATA_DIR = Path(__file__).parent / "data"

label_fields = [
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
    path = DATA_DIR / f"{field}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No reference file for field '{field}'.")
    return path.read_text().strip().split("\n")


def label_sort_key(s):
    if 'family' in s:
        return 0
    elif 'genus' in s:
        return 1
    elif 'species' in s:
        return 2
    elif 'infrasp_taxon' in s:
        return 3
    elif 'authority' in s:
        return 4
    elif 'collector_number' in s:
        return 5
    elif 'collector' in s:
        return 6
    elif 'locality' in s:
        return 7
    elif 'geolocation' in s:
        return 8
    elif 'year' in s:
        return 9
    elif 'month' in s:
        return 10
    elif 'day' in s:
        return 11
    else:
        return 12
    

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
    col_options = [ "institutional label", "id" ] + label_fields

    missing_cols = [col for col in col_options if col not in df.columns]
    df[missing_cols] = ""

    score_cols = sorted([col for col in df.columns if '_match_score' in col], key=label_sort_key)
    ocr_cols = sorted([col for col in df.columns if '_ocr_results' in col], key=label_sort_key)
    image_files_cols = sorted([col for col in df.columns if '_image' in col or 'predictions' in col], key=label_sort_key)

    cols = col_options + score_cols + ['label_classification'] + ocr_cols + image_files_cols

    extra_cols = [col for col in df.columns if col not in cols]

    cols = cols + extra_cols
    df = df[cols]
    df = df.fillna('')

    
    # CSV output
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # If the file already exists, then concatenate it
        if output_path.exists():
            old_df = pd.read_csv(output_path)
            df = pd.concat([old_df, df])
            console.print(f"Appending Hespi results to: '{output_path}'")
        else:
            console.print(f"Writing Hespi results to: '{output_path}'")
        
        df.to_csv(output_path, index=False)

    return df


def adjust_text(field:str, recognised_text:str, fuzzy:bool, fuzzy_cutoff:float, reference:Dict):
    text_adjusted = adjust_case(field, recognised_text)
    match_score = ""

    # Match with database
    if fuzzy and field in reference:
        close_matches = get_close_matches(
            text_adjusted,
            reference[field],
            cutoff=fuzzy_cutoff,
            n=1,
        )
        if close_matches:
            match_score = round(SequenceMatcher(None, text_adjusted, close_matches[0]).ratio(), 3)
            text_adjusted = close_matches[0]
        else:
            match_score = 0

    if recognised_text != text_adjusted:
        console.print(
            f"Recognized text [red]'{recognised_text}'[/red] adjusted to [purple]'{text_adjusted}'[/purple] (fuzzy match score: {match_score})"
        ) 
    else:
        if match_score != "":
            console.print(
                f"Recognized text [red]'{recognised_text}'[/red] (fuzzy match score: {match_score})"
            )    
    return (text_adjusted, match_score)


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

    