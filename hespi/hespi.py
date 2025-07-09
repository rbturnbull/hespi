import re
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
from functools import cached_property
from rich.console import Console
from collections import defaultdict
from rich.progress import track

from .yolo import yolo_output, predictions_filename
from .ocr import Tesseract, TrOCR, TrOCRSize
from .download import get_location
from .util import mk_reference, ocr_data_df, adjust_text, get_stub, ocr_data_print_tables
from .ocr import TrOCRSize
from .report import write_report
from .llm import llm_correct_detection_results, get_llm

console = Console()

DEFAULT_RELEASE_PREFIX = "https://github.com/rbturnbull/hespi/releases/download/v0.4.0"
DEFAULT_SHEET_COMPONENT_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/sheet-component.pt.gz"
DEFAULT_LABEL_FIELD_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/label-field.pt.gz"
DEFAULT_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS = f"https://github.com/rbturnbull/hespi/releases/download/v0.4.2/institutional-label-classifier.pkl.gz"


CLASSIFICATION_EMOJI = {
    "handwritten": "✍️",
    "printed": "🖨️",
    "typewriter": "⌨️",
}


class Hespi():
    def __init__(
        self,
        trocr_size: TrOCRSize = TrOCRSize.LARGE.value,
        sheet_component_weights: str = "",
        label_field_weights: str = "",
        primary_specimen_label_classifier_weights: str = "",
        force_download:bool = False,
        gpu: bool = True,
        fuzzy: bool = True,
        fuzzy_cutoff: float = 0.8,
        htr: bool = True,
        llm_model: str = "gpt-4o",
        llm_api_key: str = "",
        llm_temperature: float = 0.0,
        batch_size:int = 4,
        sheet_component_res:int = 1280,
        label_field_res:int = 1280,
    ):
        self.trocr_size = trocr_size
        self.sheet_component_weights = sheet_component_weights or DEFAULT_SHEET_COMPONENT_WEIGHTS
        self.label_field_weights = label_field_weights or DEFAULT_LABEL_FIELD_WEIGHTS
        self.primary_specimen_label_classifier_weights = primary_specimen_label_classifier_weights or DEFAULT_PRIMARY_SPECIMEN_LABEL_CLASSIFIER_WEIGHTS
        self.force_download = force_download
        self.fuzzy = fuzzy
        self.fuzzy_cutoff = fuzzy_cutoff
        self.htr = htr
        self.batch_size = batch_size
        self.sheet_component_res = sheet_component_res
        self.label_field_res = label_field_res

        if llm_model and llm_model.lower() != "none":
            self.llm = get_llm(llm_model, llm_api_key, llm_temperature)
        else:
            self.llm = None
        
        # Check if gpu is available
        import torch
        self.gpu = gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.gpu else "cpu"

    def get_yolo(self, weights_url:str) -> "YOLO":
        from ultralytics import YOLO

        weights = get_location(weights_url, force=self.force_download)
        model = YOLO(weights)
        model.to(self.device)
        return model

    @cached_property
    def sheet_component_model(self):
        model = self.get_yolo(self.sheet_component_weights)

        def replace_name(name):
            if name == "institutional label":
                return "primary specimen label"
            else:
                return name

        model.model.names = {key: replace_name(name) for key, name in model.names.items()}
        return model

    @cached_property
    def label_field_model(self):
        return self.get_yolo(self.label_field_weights)

    @cached_property
    def primary_specimen_label_classifier(self):
        from torchapp.examples.image_classifier import ImageClassifier

        model = ImageClassifier()
        model.pretrained = get_location(self.primary_specimen_label_classifier_weights, force=self.force_download)
        return model

    @cached_property
    def reference(self):
        return mk_reference()

    @cached_property
    def tesseract(self):
        return Tesseract()

    @cached_property
    def trocr(self):
        print(f"Loading TrOCR model of size '{self.trocr_size}'")
        trocr = TrOCR(size=self.trocr_size)
        return trocr

    def sheet_component_detect(
        self,
        images:List[Path],
        output_dir:Path,
    ):
        return yolo_output(
            self.sheet_component_model, 
            images, 
            output_dir=output_dir, 
            res=self.sheet_component_res,
        )

    def label_field_model_detect(
        self,
        images:List[Path],
        output_dir:Path,
    ):
        return yolo_output(
            self.label_field_model, 
            images, 
            output_dir=output_dir, 
            res=self.label_field_res,
        )

    def detect(
        self,
        images:List[Path],
        output_dir:Path,
        report:bool = True,
    ):

        # Clean up images input
        if isinstance(images, (Path, str)):
            images = [images]
        images = [get_location(image, cache_dir=output_dir/"downloads") for image in images]
        if len(images) == 1:
            console.print(f"Processing '{images[0]}'")
        else:
            console.print(f"Processing {len(images)} images")

        # Sheet-Components predictions
        component_files = self.sheet_component_detect(images, output_dir=output_dir)

        ocr_data = {}
        
        # Primary Specimen Label Field Detection Model Predictions
        for stub, components in component_files.items():
            for component in components:
                if re.match(r".*\.primary_specimen_label-?\d*.jpg$", component.name):
                    ocr_data[str(component)] = self.primary_specimen_label_detect(
                        component, 
                        stub=stub,
                        output_dir=output_dir / stub,
                    )


        df = ocr_data_df(ocr_data, output_path=output_dir/"hespi-results.csv")
        ocr_data_print_tables(df)

        # Write report
        if report:
            if len(images) == 1:
                # If report is for a single image, include in report name
                report_path = output_dir/f"hespi-report-{images[0].name}.html"
            else:
                # If report is for multiple images, make sure it is unique
                report_path = output_dir/"hespi-report.html"
                index = 1
                while report_path.exists():
                    index += 1
                    report_path = output_dir/f"hespi-report-{index}.html"
            
            write_report(report_path, component_files, df)

        return df

    def primary_specimen_label_classify(self, component:Path, classification_csv:Path) -> str:
        component = Path(component)
        console.print(f"Classifying '{component.name}':")
        classifier_results = self.primary_specimen_label_classifier(
            items=component,
            pretrained=self.primary_specimen_label_classifier.pretrained,
            gpu=self.gpu,
            output_csv=classification_csv,
            verbose=False,
        )
        # Get classification of institution label
        if isinstance(classifier_results, pd.DataFrame):
            classification = classifier_results.iloc[0]["prediction"]
        else:
            classification = str(classifier_results)

        emoji = CLASSIFICATION_EMOJI.get(classification, "")

        console.print( f"[red]{classification}[/red] {emoji}")
        console.print(f"Classification result to '{classification_csv}'")

        return classification
    
    def determine_best_ocr_result(self, detection_result, preferred_engine:str="") -> Tuple:
        assert isinstance(detection_result, list)
        best_text = ''
        best_match_score = ''
        best_engine = ''

        # If there are no results, then return empty strings
        if len(detection_result) == 0:
            return best_text, best_match_score, best_engine
        
        # If there is only one result, then return that result
        if len(detection_result) == 1:
            best_text = detection_result[0]['adjusted_text']
            best_match_score = detection_result[0]['match_score']
            # leave 'best engine' as empty if there is only one result
            return best_text, best_match_score, best_engine

        # Check if there are any results with scores
        detection_results_with_scores = [result for result in detection_result if result['match_score'] not in ['', 0]]
        if len(detection_results_with_scores) > 0:        
            best_result = max(detection_results_with_scores, key=lambda x: x['match_score'])
            best_text = best_result['adjusted_text']
            best_engine = best_result['ocr']
            best_match_score = best_result['match_score']
            return best_text, best_match_score, best_engine

        # check if we can use the preferred engine and if so, restrict results to that one
        preferred_results = [result for result in detection_result if result['ocr'] == preferred_engine]
        if len(preferred_results) > 0:
            detection_result = preferred_results

        # If there are still multiple results, then use the longest text
        if len(detection_result) > 0:
            best_text = max(detection_result, key=lambda x: len(x['adjusted_text']))['adjusted_text']

        return best_text, best_match_score, best_engine            

    def primary_specimen_label_detect(self, component, stub, output_dir) -> Dict:
        detection_results = {"id": stub}

        # Primary Specimen Label Classification
        classification = self.primary_specimen_label_classify(
            component=component,
            classification_csv = component.parent / f"label_classification.txt",
        )
        detection_results["label_classification"] = classification

        field_files = self.label_field_model_detect(
            [component],
            output_dir=output_dir,
        )

        primary_specimen_label_stub = get_stub(component)
        primary_specimen_label_dir = component.parent/primary_specimen_label_stub
        predictions_path = primary_specimen_label_dir/predictions_filename(primary_specimen_label_stub)
        detection_results["predictions"] = predictions_path

        # Text Recognition on bounding boxes found by YOLO
        for fields in field_files.values():
            for field_file in track(fields, description=f"Reading fields for {component.name}"):
                field_results = self.read_field_file(
                    field_file, 
                    classification,
                )

                for key, value in field_results.items():
                    if key not in detection_results:
                        detection_results[key] = value
                    else:
                        detection_results[key] = (
                            detection_results[key] + value 
                            if isinstance(value, list) 
                            else [detection_results[key], value]
                        )

        results = {}

        # Determining Recognised Text for fields in the reference database
        best_engine_results = []
        for key, detection_result in detection_results.items():
            if 'ocr_results' in key:
                field_name = key.replace('_ocr_results', '')
                assert isinstance(detection_result, list)
                if field_name in self.reference.keys():
                    best_text, best_match_score, best_engine = self.determine_best_ocr_result(detection_result)
                    results[field_name] = best_text
                    results[f"{field_name}_match_score"] = best_match_score
                    if best_engine:
                        best_engine_results.append(best_engine)

        # Get preferred engine from the best engine results
        if len(best_engine_results) > 0:
            from collections import Counter
            counter = Counter(best_engine_results)
            preferred_engine = counter.most_common(1)[0][0]
        elif detection_results.get('label_classification', None) in ["printed", "typewriter"]:
            preferred_engine = 'Tesseract'
        else:
            preferred_engine = 'TrOCR'

        # Determining Recognised Text for fields not in the reference database
        for key, detection_result in detection_results.items():
            if 'ocr_results' in key:
                field_name = key.replace('_ocr_results', '')
                assert isinstance(detection_result, list)
                if field_name not in self.reference.keys():
                    best_text, _, _ = self.determine_best_ocr_result(detection_result, preferred_engine=preferred_engine)
                    results[field_name] = best_text

            # splitting multiple image files into two columns
            elif 'image' in key:
                if isinstance(detection_result, list) and len(detection_result) > 1:
                    for i, image_path in enumerate(detection_result):
                        if i == 0:
                            detection_results[key] = image_path
                        else:
                            results[f"{key}_{i+1}"] = image_path
        
        detection_results.update(results)

        if self.llm:
            llm_correct_detection_results(self.llm, component, detection_results)

        return detection_results

    
    def read_field_file(
        self,
        field_file:Path,
        classification:str = None,
    ) -> Dict:
        """
        Reads the text of an image of a field from an primary specimen label.

        Args:
            field_file (Path): The path to an image file of a text field. 
                The type of field needs to be in the second last component of the filename when split by periods.
                e.g. XXXXX.FIELD_NAME.jpg
                           ^^^^^^^^^^
            classification (str, optional): The classification of the primary specimen label from the primary_specimen_label_classifier model.
                If it is 'printed' or 'typewriter' then the Tesseract OCR result will be favoured.
                Otherwise the TrOCR model result will be favoured.

        Returns:
            Dict: A dictionary with the results from TrOCR (if the `htr` flag is on), Tesseract and an adjusted form of the text,
                which changes the case depending on the field type and fuzzy matched with the reference database if `fuzzy` is requested.
        """
        field_file = Path(field_file)
        field_file_components = field_file.name.split(".")
        assert len(field_file_components) >= 2
        field = field_file_components[-2].split("-")[0]
        classification = classification or ""

        detection_results = defaultdict(list)              
        detection_results[f"{field}_image"].append(field_file)
        
        # HTR
        htr_text = ''
        if self.htr:
            htr_text = self.trocr.get_text(field_file)
            if htr_text:
                # console.print(
                #     f"Handwritten Text Recognition (TrOCR): [red]'{htr_text}'[/red]"
                # )
                
                adjusted_text, match_score = adjust_text(
                    field, 
                    htr_text, 
                    fuzzy=self.fuzzy, 
                    fuzzy_cutoff=self.fuzzy_cutoff, 
                    reference=self.reference,
                )
                
                detection_results[f"{field}_ocr_results"].append(
                    {
                        'ocr': 'TrOCR', 
                        'original_text_detected': htr_text, 
                        'adjusted_text': adjusted_text,
                        'match_score': match_score,
                    }
                )

        # OCR
        tesseract_text = self.tesseract.get_text(field_file)
        if tesseract_text:
            # console.print(
            #     f"Optical Character Recognition (Tesseract): [red]'{tesseract_text}'[/red]"
            # )            

            adjusted_text, match_score = adjust_text(
                field, 
                tesseract_text, 
                fuzzy=self.fuzzy, 
                fuzzy_cutoff=self.fuzzy_cutoff, 
                reference=self.reference,
            )
            
            detection_results[f"{field}_ocr_results"].append(
                {
                    'ocr': 'Tesseract', 
                    'original_text_detected': tesseract_text, 
                    'adjusted_text': adjusted_text, 
                    'match_score': match_score,
                }
            )            
        
        return detection_results

