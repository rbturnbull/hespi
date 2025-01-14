from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner
from hespi import Hespi

from hespi import main

runner = CliRunner()


def test_help():
    result = runner.invoke(main.app, ["--help"])
    assert result.exit_code == 0
    assert "output" in result.stdout


@patch.object(Hespi, 'detect')
def test_detect_command(mock_detect):
    result = runner.invoke(main.app, ["image1.tif", "image2.jpg", "--output-dir", "output", "--llm", "none"])
    assert result.exit_code == 0
    mock_detect.assert_called_once_with(["image1.tif", "image2.jpg"], Path("output"))

