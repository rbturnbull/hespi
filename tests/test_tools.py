from typer.testing import CliRunner
from hespi import tools
from unittest.mock import patch

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


def test_bibtex():
    result = runner.invoke(tools.app, ["bibtex"])
    assert result.exit_code == 0
    assert "@article{thompson2023_identification" in result.stdout
    assert "@misc{sheet_component_data" in result.stdout
