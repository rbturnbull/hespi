import tempfile
import pytest
from pathlib import Path

from hespi import util


def test_adjust_case():
    assert "Acanthochlamydaceae" == util.adjust_case("family", "ACANTHOCHLAMYDACEAE")
    assert "Abutilon" == util.adjust_case("genus", "abutilon")
    assert "zostericolum" == util.adjust_case("species", "Zostericolum")    
    assert "unTitled" == util.adjust_case("other", "unTitled")    


def test_read_reference_authority():
    items = util.read_reference("authority")
    assert len(items) >= 32481
    assert "(A.A.Fisch.Waldh.) Nannf." in items


def test_read_reference_genus():
    items = util.read_reference("genus")
    assert len(items) >= 13330
    assert "Abelia" in items


def test_read_reference_family():
    items = util.read_reference("family")
    assert len(items) >= 2711
    assert "Acalyphaceae" in items


def test_read_reference_species():
    items = util.read_reference("species")
    assert len(items) >= 44333
    assert "Martini" in items


def test_read_reference_unknown():
    with pytest.raises(FileNotFoundError):
        util.read_reference("location")


def test_ocr_data_df():
    required_columns = [
        'institutional label', 
        'id', 'family', 'genus', 'species',
        'infrasp_taxon', 'authority', 'collector_number', 'collector',
        'locality', 'geolocation', 'year', 'month', 'day',
        '<--results|ocr_details-->', 'image_links-->', 'ocr_results_split-->'
    ]
    df = util.ocr_data_df(
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
        df = util.ocr_data_df(
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


