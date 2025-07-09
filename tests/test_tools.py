from typer.testing import CliRunner
from hespi import tools
from unittest.mock import patch
from pathlib import Path
from .test_ocr import MockProcessor, MockModel

runner = CliRunner()


def mock_get_location(path):
    return path


def test_tools_help():
    result = runner.invoke(tools.app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: " in result.stdout


@patch("hespi.tools.get_location", mock_get_location)
def test_sheet_component_location():
    result = runner.invoke(tools.app, ["sheet-component-location"])
    assert result.exit_code == 0
    assert "hespi" in result.stdout
    assert "sheet-component" in result.stdout
    assert ".pt" in result.stdout


@patch("hespi.tools.get_location", mock_get_location)
def test_label_field_location():
    result = runner.invoke(tools.app, ["label-field-location"])
    assert result.exit_code == 0
    assert "/hespi/" in result.stdout
    assert "label-field" in result.stdout
    assert ".pt" in result.stdout


@patch("hespi.tools.get_location", mock_get_location)
def test_primary_specimen_label_classifier_location():
    result = runner.invoke(tools.app, ["primary-specimen-label-classifier-location"])
    assert result.exit_code == 0
    assert "hespi" in result.stdout
    assert "label-classifier" in result.stdout.replace("\n", "")
    assert ".pkl" in result.stdout


def test_bibtex():
    result = runner.invoke(tools.app, ["bibtex"])
    assert result.exit_code == 0
    assert "@article{thompson2023_identification" in result.stdout
    assert "@misc{sheet_component_data" in result.stdout


@patch('transformers.TrOCRProcessor.from_pretrained', lambda *args: MockProcessor() )
@patch('transformers.VisionEncoderDecoderModel.from_pretrained', lambda *args: MockModel() )
def test_trocr():
    result = runner.invoke(tools.app, ["trocr", str(Path(__file__).parent/"testdata/test.jpg")])
    assert result.exit_code == 0
    assert "recognized text" in result.stdout
