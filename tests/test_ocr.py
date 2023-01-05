from hespi import ocr
import pytest

def test_trocrsize():
    assert str(ocr.TrOCRSize.SMALL) == "small"
    assert str(ocr.TrOCRSize.BASE) == "base"
    assert str(ocr.TrOCRSize.LARGE) == "large"


def test_abstract_class():
    with pytest.raises(NotImplementedError):
        ocr.OCR().get_text("")