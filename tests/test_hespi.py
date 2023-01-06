from pathlib import Path
from hespi import Hespi
from unittest.mock import patch
from torchapp.examples.image_classifier import ImageClassifier

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

