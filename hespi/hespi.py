from typing import List, Dict
from pathlib import Path
import pandas as pd
from functools import cached_property
import torch
from torchapp.examples.image_classifier import ImageClassifier
from rich.console import Console
from ultralytics import YOLO

from .yolo import yolo_output, predictions_filename
from .ocr import Tesseract, TrOCR, TrOCRSize
from .download import get_location
from .util import read_reference, ocr_data_df, adjust_text, get_stub
from .ocr import TrOCRSize
from .report import write_report

console = Console()

DEFAULT_RELEASE_PREFIX = "https://github.com/rbturnbull/hespi/releases/download/v0.2.5"
DEFAULT_SHEET_COMPONENT_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/sheet-component-medium.pt.gz"
DEFAULT_INSTITUTIONAL_LABEL_FIELDS_WEIGHTS = f"{DEFAULT_RELEASE_PREFIX}/institutional-label-field.pt.gz"
DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS = f"https://github.com/rbturnbull/hespi/releases/download/v0.1.0-alpha/institutional-label-classifier.pkl"


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
        tmp_dir:str = None,
        batch_size:int = 4,
    ):
        self.trocr_size = trocr_size
        self.sheet_component_weights = sheet_component_weights or DEFAULT_SHEET_COMPONENT_WEIGHTS
        self.institutional_label_fields_weights = institutional_label_fields_weights or DEFAULT_INSTITUTIONAL_LABEL_FIELDS_WEIGHTS
        self.institutional_label_classifier_weights = institutional_label_classifier_weights or DEFAULT_INSTITUTIONAL_LABEL_CLASSIFIER_WEIGHTS
        self.force_download = force_download
        self.fuzzy = fuzzy
        self.fuzzy_cutoff = fuzzy_cutoff
        self.htr = htr
        self.tmp_dir = tmp_dir
        
        # Check if gpu is available
        self.gpu = gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.gpu else "cpu"

    def get_yolo(self, weights_url:str) -> YOLO:
        weights = get_location(weights_url, force=self.force_download)
        model = YOLO(weights)
        model.to(self.device)
        return model

    @cached_property
    def sheet_component_model(self):
        return self.get_yolo(self.sheet_component_weights)

    @cached_property
    def institutional_label_fields_model(self):
        return self.get_yolo(self.institutional_label_fields_weights)

    @cached_property
    def institutional_label_classifier(self):
        model = ImageClassifier()
        model.pretrained = get_location(self.institutional_label_classifier_weights, force=self.force_download)
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
        return yolo_output(self.sheet_component_model, images, output_dir=output_dir, tmp_dir_prefix=self.tmp_dir)

    def institutional_label_fields_model_detect(
        self,
        images:List[Path],
        output_dir:Path,
    ):
        return yolo_output(self.institutional_label_fields_model, images, output_dir=output_dir, tmp_dir_prefix=self.tmp_dir)

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
        console.print(f"Processing {len(images)} image(s)")

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

        df = ocr_data_df(ocr_data, output_path=output_dir/"ocr_results.csv")

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
            classification_csv = component.parent / f"label_classification.txt",
        )
        detection_results["label_classification"] = classification

        field_files = self.institutional_label_fields_model_detect(
            [component],
            output_dir=output_dir,
        )

        institutional_label_stub = get_stub(component)
        institutional_label_dir = component.parent/institutional_label_stub
        predictions_path = institutional_label_dir/predictions_filename(institutional_label_stub)
        detection_results["predictions"] = predictions_path

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
        classification:str = None,
    ) -> Dict:
        """Reads the text of an image of a field from an institutional label.

        Args:
            field_file (Path): The path to an image file of a text field. 
                The type of field needs to be in the second last component of the filename when split by periods.
                e.g. XXXXX.FIELD_NAME.jpg
                           ^^^^^^^^^^
            classification (str, optional): The classification of the institutional label from the institutional_label_classifier model.
                If it is 'printed' or 'typewriter' then the Tesseract OCR result will be favoured.
                Otherwise the TrOCR model result will be favoured.

        Returns:
            Dict: A dictionary with the results from TrOCR (if the `htr` flag is on), Tesseract and an adjusted form of the text,
                which changes the case depending on the field type and fuzzy matched with the reference database if `fuzzy` is requested.
        """
        field_file = Path(field_file)
        console.print("field_file:", field_file)
        field_file_components = field_file.name.split(".")
        assert len(field_file_components) >= 2
        field = field_file_components[-2]
        classification = classification or ""

        detection_results = {}        

            
        detection_results[f"{field}_image"] = field_file

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
            detection_results[f"{field}_Tesseract"] = tesseract_text
            console.print(
                f"Optical Character Recognition (Tesseract): [red]'{tesseract_text}'[/red]"
            )

            if classification in ["printed", "typewriter"] or not recognised_text:
                recognised_text = tesseract_text

        # Adjust text if necessary
        if recognised_text:
            detection_results[field] = adjust_text(
                field, 
                recognised_text, 
                fuzzy=self.fuzzy, 
                fuzzy_cutoff=self.fuzzy_cutoff, 
                reference=self.reference,
            )

        return detection_results

