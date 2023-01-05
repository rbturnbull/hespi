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


def test_ocr_data_df():
    required_columns = [
        "institutional label",
        "id",
        "family",
        "genus",
        "species",
        "infrasp taxon",
        "authority",
        "collector_number",
        "collector",
        "locality",
        "geolocation",
        "year",
        "month",
        "day",
    ]

    df = main.ocr_data_df(
        {
            "institutional label": {
                "family":"family",
                "id":"id",
            }
        }
    )
    assert (df.columns == required_columns).all()
    assert len(df) == 1


    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)/"out.csv"
        df = main.ocr_data_df(
            {
                "institutional label 1": {
                    "family":"family",
                    "id":"id",
                    "extra1":"extra1",
                },
                "institutional label 2": {
                    "family":"family",
                    "id":"id",
                    "extra2":"extra2",
                }
            },
            output_path = output_path,
        )
        assert (df.columns == required_columns + ["extra1", "extra2"]).all()
        assert len(df) == 2
        assert output_path.exists()


