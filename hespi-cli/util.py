from pathlib import Path
from typing import Dict
import pandas as pd
import json
import numpy as np
import string
from rich.console import Console
from difflib import get_close_matches, SequenceMatcher
from rich.table import Column, Table
import json
from math import isnan

console = Console()
structured_print = False

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


class POSIXPathEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, Path):
         return str(obj)
      return super().default(obj)

# Utility class to catch the return value of a generator after all "yields"


class Generator:
   def __init__(self, gen):
      self.gen = gen

   def __iter__(self):
      self.value = yield from self.gen
      return self.value
   
def set_structured_print(value: bool):
   global structured_print
   structured_print = value

def hprint(message: str|Table, color: str = None, overall_progress: int = -1, progress: int = -1, skip_if_structured: bool = False):
   global structured_print
   if not structured_print:
      if color is not None:
         message = f"[{color}]{message}[/{color}]"
      return console.print(message)
   if skip_if_structured:
      return
   if type(message) is Table:
      message = str(message)
   console.print(
      json.dumps({
         "message": message,
         "color": color,
         "overall_progress": overall_progress,
         "progress": progress
      })
   )

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


def clean_data_for_json(data: dict) -> dict:
   if isinstance(data, dict):
      return {key: clean_data_for_json(value) for key, value in data.items()}
   elif isinstance(data, list):
      return [clean_data_for_json(item) for item in data]
   elif isinstance(data, Path):
      return str(data)
   else:
      return data


def relative_to_output(path, output_path: Path):
   if isinstance(path, list):
      return [relative_to_output(p, output_path) for p in path]
   try:
      return str(Path(path).relative_to(output_path.parent))
   except Exception as e:
      hprint(f"Error converting path '{path}' to path relative to '{output_path.parent}': {e}")
      return path
   
def clean_prediction_data(root: dict, output_path: Path) -> dict:
   def process_field_images():
      img_paths:Path = root[key]
      if isinstance(img_paths, list):
         img_paths = [ip.absolute() for ip in img_paths]
      rel_path = relative_to_output(img_paths, output_path)
      # hprint(f"Found {field} image field ({key}).\n\t- Absolute path: {img_paths}\n\t - Relative path (to {output_path.parent}): {rel_path}")
      # hprint(f"Current clean_root[{field}]]: {clean_root[field]}")
      if 'image' not in clean_root[field]:
         clean_root[field]['image'] = {}
      clean_root[field]['image']['relative'] = [str(p) for p in rel_path] if isinstance(rel_path, list) else str(rel_path)
      clean_root[field]['image']['absolute'] = [str(p) for p in img_paths] if isinstance(img_paths, list) else str(rel_path)
      
   def process_ocr_results():
      if isinstance(root[key], list):
         clean_root[field][subfield] = [clean_data_for_json(item) for item in root[key]]
   
   clean_root = {}
   # print(f"*** Cleaning prediction data ***")
   for key in root.keys():
      # print(f"\t Processing key: {key}")
      is_field_key = False
      for field in label_fields:
         # # print(f"\t\t Key [{key}]")
         if field.lower() in key.lower():
            # # print(f"\t\t [{key}]: {key}")
            is_label_field = True
            subfield = key.replace(f"{field}_", "")
            # print(f"\t\t [{key}][{field} -> {subfield}]: ", end="")
            if field not in clean_root:
               clean_root[field] = {}
            if subfield == field:
               # print(f"Found main key of [{field}]: [{key}]")
               clean_root[field]["match_text"] = root[key]
            elif 'image' in subfield.lower():
               # print(f"Found image key!")
               process_field_images()
            elif 'ocr_result' in subfield.lower():
               # print(f"Found ocr_results {subfield}")
               process_ocr_results()
            else:
               # print(f"Found unexpected field key {subfield}")
               pass
            is_field_key = True
            break
      if not is_field_key:
         # If the key is not a label field, just copy it over
         if 'predictions' in key.lower():
            clean_root[key] = relative_to_output(Path(root[key]).absolute(), output_path)
         elif key not in clean_root:
            clean_root[key] = root[key]
   return clean_root


def clean_sheet_components(component_files: Dict, output_path: Path = None) -> Dict:
   def get_classification(path):
      return Path(path).name.split(".")[-2].replace("_", " ").title()

   if component_files is None:
      return None
   clean_components = {}
   for key, components in component_files.items():
      clean_components[key] = []
      for c in components:
         clean_components[key].append({
            "absolute_path": str(c.absolute()),
            "relative_path": relative_to_output(c.absolute(), output_path),
            "classification": get_classification(c),
         })
   return clean_components


def ocr_data_json(data: dict, component_files: Dict = None, output_path: Path = None) -> None:
   """    
   Exports a .hespi file (i.e. JSON) following the structure of the ocr_data Dict

   Args:
      data (dict): The data coming from the text recognition models. 
         The keys are the institutional labels.
         The values are dictionaries with the fields as keys and the values as the recognised text.
      component_files (Dict): A dictionary with all the files generated by running the sheet component model.
      output_path (Path, optional): Where to save a CSV file if given. Defaults to None.

   Returns:
      pd.DataFrame: The text recognition data as a Pandas dataframe
   """
   output_path = Path(output_path).absolute() if isinstance(output_path, str) else output_path.absolute()
   # Clean the component_files an extract relative paths and classification from the absolute path
   component_files = clean_sheet_components(component_files, output_path)
   skip_if_structured = True
   clean_data = {}
   hprint(f"\n------Cleaning JSON data. Output path: {output_path}--------", skip_if_structured=skip_if_structured)
   for root_key in data.keys():
      # Do not process a component_files dict if it's one of the root keys
      if "component_files" in root_key.lower():
         hprint(f"Skipping key {root_key} as it is a component file", skip_if_structured=skip_if_structured)
         continue
      hprint(f"\nCleaning JSON data for: {root_key}.\n\t-Keys: {list(data[root_key].keys())}", skip_if_structured=skip_if_structured)
      clean_root = clean_prediction_data(data[root_key], output_path)
      clean_root["label_file"] = root_key
      if component_files is not None:
         clean_root["component_files"] = component_files[clean_root["id"]]
         del component_files[clean_root["id"]]
      hprint(f"Adding key {clean_root.get('id', 'NO_ID_KEY_IN_DICT')} to new clean JSON", skip_if_structured=skip_if_structured)
      clean_data[clean_root["id"]] = clean_root

   # In case there is any component_files that was not in the ocr_data
   if component_files is not None and len(component_files) > 0:
      clean_data["_component_files"] = component_files
   with open(str(output_path), "w") as f:
      json.dump(clean_data, f, indent=3, cls=POSIXPathEncoder)


def ocr_data_df(data: dict, output_path: Path = None) -> pd.DataFrame:
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
   df = df.fillna(value="")
   df = df.reset_index().rename(columns={"index": "institutional label"})

   # Splitting the ocr_results columns into seperate original text, adjusted, and score
   # Enables the html report to pull the data
   for col in df.columns:
      if 'ocr_results' in col:
         field_name = col.replace('_ocr_results', '')
         new_columns = df[f"{field_name}_ocr_results"].apply(process_row_ocr_results, field_name=field_name).apply(pd.Series)

         df = pd.concat([df, new_columns], axis=1)

   # insert columns not included in dataframe, and re-order
   # including any columns not included in col_options to account for any updates
   col_options = ["institutional label", "id"] + label_fields

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
         hprint(f"Appending Hespi text results to: '{output_path}'", skip_if_structured=True)
      else:
         hprint(f"Writing Hespi text results to: '{output_path}'", skip_if_structured=True)
         write_df = df

      write_df.to_csv(output_path, index=False)
      
      # try:
      #    # exported_df = pd.read_csv(output_path)
      #    json_df = write_df.copy()
      #    json_df = json_df.applymap(lambda x: str(x) if isinstance(x, Path) else x)
      #    json_df.to_json(output_path.with_suffix('.full.json'), orient="records", indent=3)
      # except Exception as e:
      #    hprint(f"Error writing JSON file: {e}", "red")
   return df


def to_json(df):
   json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
   json_clean = []

   for obj in json_list:
      clean_obj = {}
      for k, v in obj.items():
         try:
               if isnan(v):
                  obj[k] = None
                  continue
         except:
               pass
         clean_obj[k] = v
      json_clean.append(clean_obj)
   with open("hespi-results-list.json", "w") as f:
      json.dump(json_list, f, indent=3)
   with open("hespi-results-clean.json", "w") as f:
      json.dump(json_clean, f, indent=3)


def ocr_result_str(row, field_name: str, ocr_type: str) -> str:
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
      filename = Path(row["institutional label"]).name
      table = Table(
         Column("Field"),
         Column("Text"),
         Column("Tesseract"),
         Column("TrOCR"),
         Column("LLM"),
         title=f"Fields in institutional label: {filename}",
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

      hprint(table)


def adjust_text(field: str, recognised_text: str, fuzzy: bool, fuzzy_cutoff: float, reference: Dict):
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


def get_stub(path: Path) -> str:
   """
   Gets the part of the filename before the last extension

   Args:
      path (Path): The path to the file

   Returns:
      str: the part of the filename before the last extension
   """
   last_period = path.name.rfind(".")
   stub = path.name[:last_period] if last_period else path.name

   # Remove any leading or trailing whitespace
   stub = stub.strip()

   # Replace punctuation with an underscore
   translate = str.maketrans(string.punctuation, "_" * len(string.punctuation))
   stub = stub.translate(translate)

   return stub
