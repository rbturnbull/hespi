import pytest
from unittest.mock import patch

from hespi import ocr

def test_trocrsize():
    assert str(ocr.TrOCRSize.SMALL) == "small"
    assert str(ocr.TrOCRSize.BASE) == "base"
    assert str(ocr.TrOCRSize.LARGE) == "large"


def test_abstract_class():
    with pytest.raises(NotImplementedError):
        ocr.OCR().get_text("")

@patch('pytesseract.image_to_string', return_value="recognized text")
def test_tesseract(mock_image_to_string):
    t = ocr.Tesseract()
    assert t.no_tesseract == False
    assert t.get_text('path') == "recognized text"
    assert t.no_tesseract == False


def raise_exception(*args):
    raise Exception("error")


@patch('pytesseract.image_to_string', raise_exception)
def test_tesseract_fail():
    t = ocr.Tesseract()
    assert t.no_tesseract == False
    assert t.get_text('path') is None
    assert t.no_tesseract == True
