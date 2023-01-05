import pytest
from typer.testing import CliRunner

from hespi import main

runner = CliRunner()


def test_help():
    result = runner.invoke(main.app, ["--help"])
    assert result.exit_code == 0
    assert "output" in result.stdout


def test_adjust_case():
    assert "Acanthochlamydaceae" == main.adjust_case("family", "ACANTHOCHLAMYDACEAE")
    assert "Abutilon" == main.adjust_case("genus", "abutilon")
    assert "zostericolum" == main.adjust_case("species", "Zostericolum")    


def test_read_reference_authority():
    items = main.read_reference("authority")
    assert len(items) == 32481
    assert "(A.A.Fisch.Waldh.) Nannf." in items

def test_read_reference_genus():
    items = main.read_reference("genus")
    assert len(items) == 13330
    assert "Abelia" in items

def test_read_reference_family():
    items = main.read_reference("family")
    assert len(items) == 2711
    assert "Acalyphaceae" in items

def test_read_reference_species():
    items = main.read_reference("species")
    assert len(items) == 44333
    assert "Martini" in items


def test_read_reference_unknown():
    with pytest.raises(FileNotFoundError):
        main.read_reference("location")