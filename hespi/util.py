from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import string
from rich.console import Console
from difflib import get_close_matches, SequenceMatcher
from rich.table import Column, Table


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

def strip_punctuation(field, value):
    punctuation_to_strip = string.punctuation.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    if field in ["genus", "family", "species"]:
        return value.strip(punctuation_to_strip).strip()
    
    return value


def read_reference(field):
    path = DATA_DIR / f"{field}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No reference file for field '{field}'.")
    return path.read_text().strip().split("\n")


def mk_reference() -> Dict:
    reference_fields = ["family", "genus", "species", "authority"]
    return {field: read_reference(field) for field in reference_fields}


def label_sort_key(s) -> int:
    base_name = s.split('_')[0]
    try:
        return label_fields.index(base_name)
    except ValueError:
        return len(label_fields)


def process_row_ocr_results(row, field_name):
    trocr_original = []
    trocr_adjusted = []
    trocr_match_score = []
    tesseract_original = []
    tesseract_adjusted = []
    tesseract_match_score = []
    llm_original = []
    llm_adjusted = []
    llm_match_score = []

    for d in row:
        if d['ocr'] == 'TrOCR':
            trocr_original.append(d['original_text_detected'])
            trocr_adjusted.append(d['adjusted_text'])
            trocr_match_score.append(d['match_score'])
        elif d['ocr'] == 'Tesseract':
            tesseract_original.append(d['original_text_detected'])
            tesseract_adjusted.append(d['adjusted_text'])
            tesseract_match_score.append(d['match_score'])
        elif d['ocr'] == 'LLM':
            llm_original.append(d['original_text_detected'])
            llm_adjusted.append(d['adjusted_text'])
            llm_match_score.append(d['match_score'])

    return {
        f"{field_name}_TrOCR_original": trocr_original,
        f"{field_name}_TrOCR_adjusted": trocr_adjusted,
        f"{field_name}_TrOCR_match_score": trocr_match_score,
        f"{field_name}_Tesseract_original": tesseract_original,
        f"{field_name}_Tesseract_adjusted": tesseract_adjusted,
        f"{field_name}_Tesseract_match_score": tesseract_match_score,
        f"{field_name}_LLM_original": llm_original,
        f"{field_name}_LLM_adjusted": llm_adjusted,
        f"{field_name}_LLM_match_score": llm_match_score,
    }


def flatten_single_item_lists(lst):
    if isinstance(lst, list):
        if len(lst) == 1:
            return lst[0]
        elif len(lst) == 0:
            return ''
    return lst


def ocr_data_df(data: dict, output_path: Path=None) -> pd.DataFrame:
    """    
    Creates a DataFrame of data, sorts columns and outputs a CSV with OCR values.

    Args:
        data (dict): The data coming from the text recognition models. 
            The keys are the primary specimen labels.
            The values are dictionaries with the fields as keys and the values as the recognised text.
        output_path (Path, optional): Where to save a CSV file if given. Defaults to None.

    Returns:
        pd.DataFrame: The text recognition data as a Pandas dataframe
    """
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.fillna(value="")
    df = df.reset_index().rename(columns={"index": "primary specimen label"})
    
    # Splitting the ocr_results columns into seperate original text, adjusted, and score
    # Enables the html report to pull the data 
    for col in df.columns:
        if 'ocr_results' in col:
            field_name = col.replace('_ocr_results', '')
            new_columns = df[f"{field_name}_ocr_results"].apply(process_row_ocr_results, field_name=field_name).apply(pd.Series)

            df = pd.concat([df, new_columns], axis=1)
        
    # insert columns not included in dataframe, and re-order
    # including any columns not included in col_options to account for any updates
    col_options = [ "primary specimen label", "id" ] + label_fields

    missing_cols = [col for col in col_options if col not in df.columns]
    df[missing_cols] = ""

    # creating break columns
    df['<--results|ocr_details-->'] = '    |    '
    df['image_links-->'] = '    |    '
    df['ocr_results_split-->'] = '    |    '

    # grouping other columns 
    score_cols = sorted([col for col in df.columns if '_match_score' in col and 'Tesseract' not in col and 'TrOCR' not in col], key=label_sort_key)
    ocr_cols = ['<--results|ocr_details-->'] + sorted([col for col in df.columns if '_ocr_results' in col], key=label_sort_key)
    image_files_cols = ['image_links-->'] + sorted([col for col in df.columns if '_image' in col or 'predictions' in col], key=label_sort_key)
    result_cols = ['ocr_results_split-->'] + sorted([col for col in df.columns if 'Tesseract' in col or 'TrOCR' in col], key=label_sort_key)
    
    cols = col_options + score_cols + ['label_classification'] + ocr_cols + result_cols + image_files_cols 
    extra_cols = [col for col in df.columns if col not in cols]

    cols = cols + extra_cols
    
    # Filter for only the columns that are in the dataframe
    cols = [col for col in cols if col in df.columns]
    
    df = df[cols]
    df = df.fillna('')

    # flattening all the lists so that if only one item, the list is removed, if no items, list is replaced with an empty string
    for col in df.columns:
        df[col] = df[col].apply(flatten_single_item_lists)

    # CSV output
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # If the file already exists, then concatenate it
        if output_path.exists():
            old_df = pd.read_csv(output_path)
            write_df = pd.concat([old_df, df])
            console.print(f"Appending Hespi text results to: '{output_path}'")
        else:
            console.print(f"Writing Hespi text results to: '{output_path}'")
            write_df = df
        
        write_df.to_csv(output_path, index=False)

    return df


def ocr_result_str(row, field_name:str, ocr_type:str) -> str:
    def get_list(row, key):
        result = row.get(key, "")
        if not isinstance(result, list):
            result = [result]

        return result

    original_list = get_list(row, f"{field_name}_{ocr_type}_original")
    adjusted_list = get_list(row, f"{field_name}_{ocr_type}_adjusted")
    match_score_list = get_list(row, f"{field_name}_{ocr_type}_match_score")

    assert len(original_list) == len(adjusted_list) == len(match_score_list)

    texts = []
    for original, adjusted, match_score in zip(original_list, adjusted_list, match_score_list):
        text = original
        if adjusted and adjusted != original:
            text += f" → {adjusted}"
            
        if match_score:
            text += f" ({match_score})"
        
        texts.append(text)
        
    return " | ".join(texts)


def ocr_data_print_tables(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        filename = Path(row["primary specimen label"]).name
        table = Table(
            Column("Field"),
            Column("Text"),
            Column("Tesseract"),
            Column("TrOCR"),
            Column("LLM"),
            title=f"Fields in primary specimen label: {filename}",
        )

        for field in label_fields:
            if f"{field}_image" in row and row[f"{field}_image"]:
                table.add_row(
                    field,
                    row[field],
                    ocr_result_str(row, field, ocr_type="Tesseract"),
                    ocr_result_str(row, field, ocr_type="TrOCR"),
                    ocr_result_str(row, field, ocr_type="LLM"),
                )

        console.print(table)


def adjust_text(field:str, recognised_text:str, fuzzy:bool, fuzzy_cutoff:float, reference:Dict):
    text_stripped = strip_punctuation(field, recognised_text)
    text_adjusted = adjust_case(field, text_stripped)
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
    stub = stub.replace(" ", "_").replace(".", "_").replace(":", "_").strip()
    return stub

    