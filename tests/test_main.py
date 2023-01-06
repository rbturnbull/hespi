import tempfile
import pytest
from typer.testing import CliRunner
from pathlib import Path

from hespi import main

runner = CliRunner()


def test_help():
    result = runner.invoke(main.app, ["--help"])
    assert result.exit_code == 0
    assert "output" in result.stdout

