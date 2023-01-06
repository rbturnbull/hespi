from typing import List, Dict
from pathlib import Path
import pandas as pd
from functools import cached_property
from difflib import get_close_matches
import torch
from yolov5 import YOLOv5
from torchapp.examples.image_classifier import ImageClassifier
from rich.console import Console

from .yolo import yolo_output
from .ocr import Tesseract, TrOCR, TrOCRSize
from .download import get_weights
from .util import read_reference, ocr_data_df, adjust_case
from .ocr import TrOCRSize

console = Console()

DEFAULT_RELEASE_PREFIX = "https://github.com/rbturnbull/hespi/releases/download/v0.1.0-alpha"
DEFAULT_SHEET_COMPONENT_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/sheet-component-weights.pt"
DEFAULT_INSTITUTIONAL_LABEL_FIELDS_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/institutional-label-fields.pt"
DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/institutional-label-classifier.pkl"


class Hespi():
    def __init__(
        self,
        trocr_size: TrOCRSize = TrOCRSize.BASE.value,
        sheet_component_weights: str = "",
        institutional_label_fields_weights: str = "",
        institutional_label_classifier_weights: str = "",
        force_download:bool = False,
        gpu: bool = True,
        fuzzy: bool = True,
        fuzzy_cutoff: float = 0.8,
        htr: bool = True,
    ):
        self.trocr_size = trocr_size
        self.sheet_component_weights = sheet_component_weights or DEFAULT_SHEET_COMPONENT_WEIGHTS
        self.institutional_label_fields_weights = institutional_label_fields_weights or DEFAULT_INSTITUTIONAL_LABEL_FIELDS_WEIGHTS
        self.institutional_label_classifier_weights = institutional_label_classifier_weights or DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS
        self.force_download = force_download
        self.fuzzy = fuzzy
        self.fuzzy_cutoff = fuzzy_cutoff
        self.htr = htr
        
        # Check if gpu is available
        self.gpu = gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.gpu else "cpu"

    def get_yolo(self, weights_url:str) -> YOLOv5:
        weights = get_weights(weights_url, force=self.force_download)
        return YOLOv5(weights, self.device)

    @cached_property
    def sheet_component_model(self):
        return self.get_yolo(self.sheet_component_weights)

    @cached_property
    def institutional_label_fields_model(self):
        return self.get_yolo(self.institutional_label_fields_weights)

    @cached_property
    def institutional_label_classifier(self):
        model = ImageClassifier()
        model.pretrained = get_weights(self.institutional_label_classifier_weights, force=self.force_download)
        return model

    @cached_property
    def reference(self):
        reference_fields = ["family", "genus", "species", "authority"]
        return {field: read_reference(field) for field in reference_fields}

    @cached_property
    def tesseract(self):
        return Tesseract()

    @cached_property
    def trocr(self):
        print(f"Getting TrOCR model of size '{self.trocr_size}'")
        trocr = TrOCR(size=self.trocr_size)
        print("TrOCR model available")
        return trocr

    def sheet_component_detect(
        self,
        images:List[Path],
        output_dir:Path,
    ):
        return yolo_output(self.sheet_component_model, images, output_dir=output_dir)

    def institutional_label_fields_model_detect(
        self,
        images:List[Path],
        output_dir:Path,
    ):
        return yolo_output(self.institutional_label_fields_model, images, output_dir=output_dir)

    def detect(
        self,
        images:List[Path],
        output_dir:Path,
    ):
        console.print(f"Processing {images}")

        # Sheet-Components predictions
        component_files = self.sheet_component_detect(images, output_dir=output_dir)

        ocr_data = {}
        
        # Institutional Label Field Detection Model Predictions
        for stub, components in component_files.items():
            for component in components:
                if component.name.endswith("institutional_label.jpg"):
                    ocr_data[str(component)] = self.institutional_label_detect(
                        component, 
                        stub=stub,
                        output_dir=output_dir / stub,
                    )

        return ocr_data_df(ocr_data, output_path=output_dir/"ocr_results.csv")

    def institutional_label_classify(self, component:Path, classification_csv:Path) -> str:
        console.print(f"Classifying institution label: '{component}'")
        console.print(f"Saving result to '{classification_csv}'")
        classifier_results = self.institutional_label_classifier(
            items=component,
            pretrained=self.institutional_label_classifier.pretrained,
            gpu=self.gpu,
            output_csv=classification_csv,
            verbose=False,
        )
        # Get classification of institution label
        if isinstance(classifier_results, pd.DataFrame):
            classification = classifier_results.iloc[0]["prediction"]
            console.print(
                f"'{component}' classified as '[red]{classification}[/red]'."
            )
        else:
            console.print(
                f"Could not get classification of institutional label '{component}'"
            )
            classification = ""

        return classification

    def institutional_label_detect(self, component, stub, output_dir) -> Dict:
        detection_results = {"id": stub}

        # Institutional Label Classification
        classification = self.institutional_label_classify(
            component=component,
            classification_csv = component.parent / f"{component.name[:3]}.classification.csv", # hack
        )
        detection_results["label_classification"] = classification

        field_files = self.institutional_label_fields_model_detect(
            [component],
            output_dir=output_dir,
        )

        # Text Recognition on bounding boxes found by YOLO
        for fields in field_files.values():
            for field_file in fields:
                field_results = self.read_field_file(
                    field_file, 
                    classification,
                )
                detection_results.update(field_results)
        
        return detection_results

    def read_field_file(
        self,
        field_file:Path,
        classification:str,
    ) -> Dict:
        console.print("field_file:", field_file)
        field_file_components = field_file.name.split(".")
        assert len(field_file_components) >= 5
        field = field_file_components[-2]
        classification = classification or ""

        detection_results = {}        

        # HTR
        recognised_text = ""
        if self.htr:
            htr_text = self.trocr.get_text(field_file)
            if htr_text:
                print(f"HTR: {htr_text}")
                console.print(
                    f"Handwritten Text Recognition (TrOCR): [red]'{htr_text}'[/red]"
                )

                detection_results[f"{field}_TrOCR"] = htr_text
                recognised_text = htr_text

        # OCR
        tesseract_text = self.tesseract.get_text(field_file)
        if tesseract_text:
            detection_results[f"{field}_tesseract"] = tesseract_text
            console.print(
                f"Optical Character Recognition (Tesseract): [red]'{tesseract_text}'[/red]"
            )

            if classification in ["printed", "typewriter"] or not recognised_text:
                recognised_text = tesseract_text

        # Adjust text if necessary
        if recognised_text:
            # Adjust case
            text_adjusted = adjust_case(field, recognised_text)

            # Match with database
            if self.fuzzy and field in self.reference:
                close_matches = get_close_matches(
                    text_adjusted,
                    self.reference[field],
                    cutoff=self.fuzzy_cutoff,
                    n=1,
                )
                if close_matches:
                    text_adjusted = close_matches[0]

            if recognised_text != text_adjusted:
                console.print(
                    f"Recognized text [red]'{recognised_text}'[/red] adjusted to [purple]'{text_adjusted}'[/purple]"
                )    
    
            detection_results[field] = text_adjusted

        return detection_results

