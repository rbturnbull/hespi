import tempfile
from pathlib import Path
from typing import Dict
from hespi import Hespi
import pandas as pd
from unittest.mock import patch
from torchapp.examples.image_classifier import ImageClassifier
from hespi.ocr import Tesseract, TrOCR, TrOCRSize, OCR

from .test_ocr import MockProcessor, MockModel
from .test_yolo import MockYoloModel

test_data_dir = Path(__file__).parent/"testdata"


class DictionaryPathMocker():
    def __init__(self, dictionary:Dict=None):
        dictionary = dictionary or {}
        self.dictionary = {Path(key):value for key, value in dictionary.items()}
        self.get_text_called_args = []
        
    def get_value(self, path: Path) -> str:
        self.get_text_called_args.append(path)
        return self.dictionary.get(Path(path), "")


class MockOCR(DictionaryPathMocker, OCR):
    def get_text(self, image_path: Path) -> str:
        return self.get_value(image_path)


class MockClassifier(DictionaryPathMocker, OCR):
    pretrained = ""

    def __call__(self, items, **kwargs) -> str:
        return self.get_value(items)


@patch('hespi.hespi.YOLO')
def test_get_yolo(mock):
    hespi = Hespi()
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi.get_yolo(weights)
    mock.assert_called_once_with(weights)


@patch('hespi.hespi.YOLO')
def test_sheet_component_model(mock):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        sheet_component_weights=weights
    )
    _ = hespi.sheet_component_model
    mock.assert_called_once_with(weights)


@patch('hespi.hespi.YOLO')
def test_label_field_model(mock):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    hespi = Hespi(
        label_field_weights=weights
    )
    _ = hespi.label_field_model
    mock.assert_called_once_with(weights)


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



@patch('hespi.hespi.yolo_output')
def test_sheet_component_detect(mock_yolo_output):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    mock_yolo_model = MockYoloModel()
    with patch('hespi.hespi.YOLO', return_value=mock_yolo_model) as mock_yolo_class:
        hespi = Hespi(
            sheet_component_weights=weights
        )
        hespi.sheet_component_detect("images", "output_dir")
        mock_yolo_class.assert_called_once()
        mock_yolo_output.assert_called_once_with(mock_yolo_model, "images", output_dir="output_dir", tmp_dir_prefix=None, res=1280, batch_size=4)
    


@patch('hespi.hespi.yolo_output')
def test_label_field_model_detect(mock_yolo_output):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    mock_yolo_model = MockYoloModel()
    with patch('hespi.hespi.YOLO', return_value=mock_yolo_model) as mock_yolo_class:
        hespi = Hespi(
            label_field_weights=weights
        )
        hespi.label_field_model_detect("images", "output_dir")
        mock_yolo_class.assert_called_once()
        mock_yolo_output.assert_called_once_with(mock_yolo_model, "images", output_dir="output_dir", tmp_dir_prefix=None, res=1280, batch_size=4)


def test_read_field_file_tesseract_only():
    hespi = Hespi(htr=False, fuzzy=True)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)

    assert len(result.keys()) == 2
    assert len(result["species_image"]) == 1
    assert result["species_image"][0] == image
    assert len(result['species_ocr_results']) == 1
    assert result["species_ocr_results"][0]['ocr'] == '_Tesseract'
    assert result["species_ocr_results"][0]['original_text_detected'] == 'zOstericolaXX'
    assert result["species_ocr_results"][0]['adjusted_text'] == 'zostericola'
    assert result["species_ocr_results"][0]['match_score'] == 0.917


def test_read_field_file_htr():
    hespi = Hespi(htr=True, fuzzy=False)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)

    assert len(result.keys()) == 2
    assert len(result["species_image"]) == 1
    assert result["species_image"][0] == image
    assert len(result["species_ocr_results"]) == 2

    assert result["species_ocr_results"][0]['ocr'] == '_TrOCR'
    assert result["species_ocr_results"][0]['original_text_detected'] == 'zostericolaX'
    assert result["species_ocr_results"][0]['adjusted_text'] == 'zostericolax'
    assert result["species_ocr_results"][0]['match_score'] == ""

    assert result["species_ocr_results"][1]['ocr'] == '_Tesseract'
    assert result["species_ocr_results"][1]['original_text_detected'] == 'zOstericolaXX'
    assert result["species_ocr_results"][1]['adjusted_text'] == 'zostericolaxx'
    assert result["species_ocr_results"][1]['match_score'] == ""


def test_read_field_file_fuzzy():
    hespi = Hespi(htr=True, fuzzy=True)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)

    assert len(result.keys()) == 2
    assert len(result["species_image"]) == 1
    assert result["species_image"][0] == image
    assert len(result["species_ocr_results"]) == 2

    assert result["species_ocr_results"][0]['ocr'] == '_TrOCR'
    assert result["species_ocr_results"][0]['original_text_detected'] == 'zostericolaX'
    assert result["species_ocr_results"][0]['adjusted_text'] == 'zostericola'
    assert result["species_ocr_results"][0]['match_score'] == 0.957

    assert result["species_ocr_results"][1]['ocr'] == '_Tesseract'
    assert result["species_ocr_results"][1]['original_text_detected'] == 'zOstericolaXX'
    assert result["species_ocr_results"][1]['adjusted_text'] == 'zostericola'
    assert result["species_ocr_results"][1]['match_score'] == 0.917


def test_institutional_label_classify():
    hespi = Hespi()
    targets = ["typewritten", "printed", "handwritten", ""]
    mapper = {}
    for t in targets:
        mapper[f"{t}.jpg"] = pd.DataFrame([dict(prediction=t)]) if t else ""

    hespi.institutional_label_classifier = MockClassifier(mapper)
    for path in mapper.keys():
        result = hespi.institutional_label_classify(path, "csv_output") 
        assert result == path.split('.')[0]


@patch('hespi.hespi.yolo_output', return_value={"species":["species.jpg"]})
def test_institutional_label_detect(mock_yolo_output):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    mock_yolo_model = MockYoloModel()
    with patch('hespi.hespi.YOLO', return_value=mock_yolo_model) as mock_yolo_class:
        hespi = Hespi(
            label_field_weights=weights
        )
        hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
        hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
        filename = "institution_label.jpg"
        hespi.institutional_label_classifier = MockClassifier({filename:"printed"})

        result = hespi.institutional_label_detect(Path(filename), "stub", "output_dir")
        assert result["label_classification"] == "printed"
        assert len(result["species_image"]) == 1
        assert len(result["species_ocr_results"]) == 2

        assert result["species_ocr_results"][0]['ocr'] == '_TrOCR'
        assert result["species_ocr_results"][0]['original_text_detected'] == 'zostericolaX'
        assert result["species_ocr_results"][0]['adjusted_text'] == 'zostericola'
        assert result["species_ocr_results"][0]['match_score'] == 0.957

        assert result["species_ocr_results"][1]['ocr'] == '_Tesseract'
        assert result["species_ocr_results"][1]['original_text_detected'] == 'zOstericolaXX'
        assert result["species_ocr_results"][1]['adjusted_text'] == 'zostericola'
        assert result["species_ocr_results"][1]['match_score'] == 0.917
        mock_yolo_class.assert_called_once()
        mock_yolo_output.assert_called_once_with(mock_yolo_model, [Path(filename)], output_dir="output_dir", tmp_dir_prefix=None, res=1280, batch_size=4)


@patch.object(Hespi, 'sheet_component_detect', return_value={
    "stub": [Path("stub.institutional_label.jpg"), Path("stub.swatch.jpg"), ],
})
@patch.object(Hespi, 'institutional_label_detect')
@patch('hespi.hespi.ocr_data_df')
def test_detect(mock_ocr_data_df, mock_institutional_label_detect, mock_sheet_component_detect):
    hespi = Hespi()
    with tempfile.TemporaryDirectory() as tmpdir:        
        output_dir = Path(tmpdir)
        hespi.detect(test_data_dir/"test2.jpg", output_dir)
        mock_sheet_component_detect.assert_called_once_with([test_data_dir/"test2.jpg"], output_dir=output_dir)
        mock_institutional_label_detect.assert_called_once_with(
            Path("stub.institutional_label.jpg"), 
            stub="stub",
            output_dir=output_dir/"stub")
        mock_ocr_data_df.assert_called_once()
        report_file = output_dir/"hespi-report-test2.jpg.html"
        assert report_file.exists()
        assert "<head>" in report_file.read_text()


@patch.object(Hespi, 'sheet_component_detect', return_value={
    "long-name-that-needs-to-be-truncated": [Path("long-name-that-needs-to-be-truncated.institutional_label.jpg"), Path("long-name-that-needs-to-be-truncated.swatch.jpg"), ],    
})
@patch.object(Hespi, 'institutional_label_detect')
@patch('hespi.hespi.ocr_data_df')
def test_detect_truncated(mock_ocr_data_df, mock_institutional_label_detect, mock_sheet_component_detect):
    hespi = Hespi()
    with tempfile.TemporaryDirectory() as tmpdir:        
        output_dir = Path(tmpdir)
        hespi.detect(test_data_dir/"test2.jpg", output_dir)
        mock_sheet_component_detect.assert_called_once_with([test_data_dir/"test2.jpg"], output_dir=output_dir)
        mock_institutional_label_detect.assert_called_once_with(
            Path("long-name-that-needs-to-be-truncated.institutional_label.jpg"), 
            stub="long-name-that-needs-to-be-truncated",
            output_dir=output_dir/"long-name-that-needs-to-be-truncated")
        mock_ocr_data_df.assert_called_once()
        report_file = output_dir/"hespi-report-test2.jpg.html"
        assert report_file.exists()
        assert "long-name-that-needs-to-be-tru..." in report_file.read_text()


@patch.object(Hespi, 'sheet_component_detect', return_value={
    "stub": [Path("stub.institutional_label.jpg"), Path("stub.swatch.jpg"), ],
})
@patch.object(Hespi, 'institutional_label_detect')
@patch('hespi.hespi.ocr_data_df')
def test_detect_multi(mock_ocr_data_df, mock_institutional_label_detect, mock_sheet_component_detect):
    hespi = Hespi()
    with tempfile.TemporaryDirectory() as tmpdir:        
        output_dir = Path(tmpdir)
        hespi.detect([test_data_dir/"test2.jpg", test_data_dir/"test.jpg"], output_dir)
        mock_sheet_component_detect.assert_called_once_with([test_data_dir/"test2.jpg", test_data_dir/"test.jpg"], output_dir=output_dir)
        mock_institutional_label_detect.assert_called_with(
            Path("stub.institutional_label.jpg"), 
            stub="stub",
            output_dir=output_dir/"stub")
        mock_ocr_data_df.assert_called_once()
        report_file = output_dir/"hespi-report.html"
        assert report_file.exists()
        assert "<head>" in report_file.read_text()


@patch.object(Hespi, 'sheet_component_detect', return_value={
    "stub": [Path("stub.institutional_label.jpg"), Path("stub.swatch.jpg"), ],
})
@patch.object(Hespi, 'institutional_label_detect')
@patch('hespi.hespi.ocr_data_df')
def test_detect_multi_report_exists(mock_ocr_data_df, mock_institutional_label_detect, mock_sheet_component_detect):
    hespi = Hespi()
    with tempfile.TemporaryDirectory() as tmpdir:        
        output_dir = Path(tmpdir)
        first_report_file = output_dir/"hespi-report.html"
        first_report_file.write_text("DUMMY REPORT")

        hespi.detect([test_data_dir/"test2.jpg", test_data_dir/"test.jpg"], output_dir)
        mock_sheet_component_detect.assert_called_once_with([test_data_dir/"test2.jpg", test_data_dir/"test.jpg"], output_dir=output_dir)
        mock_institutional_label_detect.assert_called_with(
            Path("stub.institutional_label.jpg"), 
            stub="stub",
            output_dir=output_dir/"stub")
        mock_ocr_data_df.assert_called_once()
        
        report_file = output_dir/"hespi-report-2.html"
        assert report_file.exists()
        assert "<head>" in report_file.read_text()


def test_determine_best_ocr_result_non_reference():
    hespi = Hespi(htr=True, fuzzy=True)
    image = Path("location.jpg")
    hespi.trocr = MockOCR({"location.jpg":"Queenscliff"})
    hespi.tesseract = MockOCR({"location.jpg":"Queeadfafdnljk"})
    
    result = hespi.read_field_file(image)
    best_text, best_match_score, best_engine = hespi.determine_best_ocr_result(result['location_ocr_results'])
    assert best_text == "Queeadfafdnljk"
    assert best_match_score == ""
    assert best_engine == ""

    best_text, best_match_score, best_engine = hespi.determine_best_ocr_result(result['location_ocr_results'], preferred_engine="_TrOCR")
    assert best_text == "Queenscliff"
    assert best_match_score == ""
    assert best_engine == ""


def test_determine_best_ocr_result_reference():
    hespi = Hespi(htr=True, fuzzy=True)
    image = Path("species.jpg")
    hespi.trocr = MockOCR({"species.jpg":"zostericolaX"})
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)
    best_text, best_match_score, best_engine = hespi.determine_best_ocr_result(result['species_ocr_results'])
    assert best_text == "zostericola"
    assert best_match_score == 0.957
    assert best_engine == "_TrOCR"

    best_text, best_match_score, best_engine = hespi.determine_best_ocr_result(result['species_ocr_results'], preferred_engine="_Tesseract")
    assert best_text == "zostericola"
    assert best_match_score == 0.957
    assert best_engine == "_TrOCR"


def test_determine_best_ocr_result_single():
    hespi = Hespi(htr=False, fuzzy=True)
    image = Path("species.jpg")
    hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX"})
    
    result = hespi.read_field_file(image)
    best_text, best_match_score, best_engine = hespi.determine_best_ocr_result(result['species_ocr_results'])
    assert best_text == "zostericola"
    assert best_match_score == 0.917
    assert best_engine == ""


def test_determine_best_ocr_result_empty():
    hespi = Hespi(htr=False, fuzzy=True)
    best_text, best_match_score, best_engine = hespi.determine_best_ocr_result([])
    assert best_text == ""
    assert best_match_score == ""
    assert best_engine == ""


@patch('hespi.hespi.yolo_output', return_value={"species":["species.jpg"], "location":["location.jpg"]})
def test_institutional_label_detect_trocr_found_preferred(mock_yolo_output):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    mock_yolo_model = MockYoloModel()
    with patch('hespi.hespi.YOLO', return_value=mock_yolo_model) as mock_yolo_class:
        hespi = Hespi(
            label_field_weights=weights
        )
        hespi.trocr = MockOCR({"species.jpg":"zostericolaX", "location.jpg":"Queenscliff"})
        hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX", "location.jpg":"Queeadfafdnljk"})
        filename = "institution_label.jpg"
        hespi.institutional_label_classifier = MockClassifier({filename:"printed"})

        result = hespi.institutional_label_detect(Path(filename), "stub", "output_dir")
        assert result["label_classification"] == "printed"
        assert result["location"] == "Queenscliff"
        assert result["species"] == "zostericola"



@patch('hespi.hespi.yolo_output', return_value={"location":["location.jpg"]})
def test_institutional_label_detect_tesseract_preferred(mock_yolo_output):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    mock_yolo_model = MockYoloModel()
    with patch('hespi.hespi.YOLO', return_value=mock_yolo_model) as mock_yolo_class:
        hespi = Hespi(
            label_field_weights=weights
        )
        hespi.trocr = MockOCR({"species.jpg":"zostericolaX", "location.jpg":"Queenscliff"})
        hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX", "location.jpg":"Queeadfafdnljk"})
        filename = "institution_label.jpg"
        hespi.institutional_label_classifier = MockClassifier({filename:"printed"})

        result = hespi.institutional_label_detect(Path(filename), "stub", "output_dir")
        assert result["label_classification"] == "printed"
        assert result["location"] == "Queeadfafdnljk"



@patch('hespi.hespi.yolo_output', return_value={"location":["location.jpg"]})
def test_institutional_label_detect_trocr_handwritten(mock_yolo_output):
    weights = test_data_dir/"test-no-extension-ebaa296923904111a3972b54eba5cf5f.dat"
    mock_yolo_model = MockYoloModel()
    with patch('hespi.hespi.YOLO', return_value=mock_yolo_model) as mock_yolo_class:
        hespi = Hespi(
            label_field_weights=weights
        )
        hespi.trocr = MockOCR({"species.jpg":"zostericolaX", "location.jpg":"Queenscliff"})
        hespi.tesseract = MockOCR({"species.jpg":"zOstericolaXX", "location.jpg":"Queeadfafdnljk"})
        filename = "institution_label.jpg"
        hespi.institutional_label_classifier = MockClassifier({filename:"handwritten"})

        result = hespi.institutional_label_detect(Path(filename), "stub", "output_dir")
        assert result["label_classification"] == "handwritten"
        assert result["location"] == "Queenscliff"



