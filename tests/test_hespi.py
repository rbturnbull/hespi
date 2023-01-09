from pathlib import Path
from typing import Dict
from hespi import Hespi
from unittest.mock import patch
from torchapp.examples.image_classifier import ImageClassifier
from hespi.ocr import Tesseract, TrOCR, TrOCRSize, OCR

from .test_ocr import MockProcessor, MockModel

test_data_dir = Path(__file__).parent/"testdata"

@patch('hespi.hespi.YOLOv5')
def test_get_yolo(mock):
    hespi = Hespi()
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi.get_yolo(weights)
    mock.assert_called_once_with(weights, hespi.device)


@patch('hespi.hespi.YOLOv5')
def test_sheet_component_model(mock):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        sheet_component_weights=weights
    )
    _ = hespi.sheet_component_model
    mock.assert_called_once_with(weights, hespi.device)


@patch('hespi.hespi.YOLOv5')
def test_institutional_label_fields_model(mock):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        institutional_label_fields_weights=weights
    )
    _ = hespi.institutional_label_fields_model
    mock.assert_called_once_with(weights, hespi.device)


def test_institutional_label_classifier():
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        institutional_label_classifier_weights=weights
    )
    model = hespi.institutional_label_classifier
    assert isinstance(model, ImageClassifier)
    assert model.pretrained == weights


def test_reference():
    hespi = Hespi()
    assert list(hespi.reference.keys()) == ["family", "genus", "species", "authority"]


def test_tesseract():
    hespi = Hespi()
    assert isinstance(hespi.tesseract, Tesseract)


@patch('transformers.TrOCRProcessor.from_pretrained', lambda *args: MockProcessor() )
@patch('transformers.VisionEncoderDecoderModel.from_pretrained', lambda *args: MockModel() )
def test_trocr():
    hespi = Hespi()
    assert isinstance(hespi.trocr, TrOCR)


@patch('hespi.hespi.YOLOv5', return_value="sheet_component_model")
@patch('hespi.hespi.yolo_output')
def test_sheet_component_detect(mock_yolo_output, mock_yolo):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        sheet_component_weights=weights
    )
    hespi.sheet_component_detect("images", "output_dir")
    mock_yolo_output.assert_called_once_with("sheet_component_model", "images", output_dir="output_dir")
    


@patch('hespi.hespi.YOLOv5', return_value="institutional_label_fields_model")
@patch('hespi.hespi.yolo_output')
def test_institutional_label_fields_model_detect(mock_yolo_output, mock_yolo):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        institutional_label_fields_weights=weights
    )
    hespi.institutional_label_fields_model_detect("images", "output_dir")
    mock_yolo_output.assert_called_once_with("institutional_label_fields_model", "images", output_dir="output_dir")


class MockOCR(OCR):
    def __init__(self, dictionary:Dict):
        self.dictionary = {Path(key):value for key, value in dictionary.items()}
        self.get_text_called_args = []

    def get_text(self, image_path: Path) -> str:
        self.get_text_called_args.append(image_path)
        return self.dictionary.get(Path(image_path), "")


def test_read_field_file_tesseract_only():
    hespi = Hespi(htr=False, fuzzy=True)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)

    assert len(result.keys()) == 2
    assert result["species_Tesseract"] == "zOstericolaXX"
    assert result["species"] == "zostericola"

def test_read_field_file_htr():
    hespi = Hespi(htr=True, fuzzy=False)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)

    assert len(result.keys()) == 3
    assert result["species_TrOCR"] == "zostericolaX"
    assert result["species_Tesseract"] == "zOstericolaXX"
    assert result["species"] == "zostericolax"    


def test_read_field_file_fuzzy():
    hespi = Hespi(htr=True, fuzzy=True)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)

    assert len(result.keys()) == 3
    assert result["species_TrOCR"] == "zostericolaX"
    assert result["species_Tesseract"] == "zOstericolaXX"
    assert result["species"] == "zostericola"        