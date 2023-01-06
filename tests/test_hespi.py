from pathlib import Path
from hespi import Hespi
from unittest.mock import patch
from torchapp.examples.image_classifier import ImageClassifier
from hespi.ocr import Tesseract, TrOCR, TrOCRSize

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
