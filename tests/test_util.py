import numpy as np
import tempfile
import pytest
from pathlib import Path

from hespi import util


def test_adjust_case():
    assert "Acanthochlamydaceae" == util.adjust_case("family", "ACANTHOCHLAMYDACEAE")
    assert "Abutilon" == util.adjust_case("genus", "abutilon")
    assert "zostericolum" == util.adjust_case("species", "Zostericolum")    
    assert "unTitled" == util.adjust_case("other", "unTitled")    


def test_adjust_text():
    reference = util.mk_reference()
    text, score = util.adjust_text("species", "zostericolumzos", True, 0.8, reference)    
    assert text == "zostericolum"
    assert score == 0.889

    text, score = util.adjust_text("species", "zostericolumzostericolum", True, 0.8, reference)    
    assert text == "zostericolumzostericolum"
    assert score == 0


def test_label_sort_key():
    assert util.label_sort_key("family") == 0
    assert util.label_sort_key("genus") == 1
    assert util.label_sort_key("species") == 2
    assert util.label_sort_key("other") == 12


def test_read_reference_authority():
    items = util.read_reference("authority")
    assert len(items) >= 269999
    assert "(A.A.Fisch.Waldh.) Nannf." in items


def test_read_reference_genus():
    items = util.read_reference("genus")
    assert len(items) >= 47591
    assert "Abelia" in items


def test_read_reference_family():
    items = util.read_reference("family")
    assert len(items) >= 3182
    assert "Acalyphaceae" in items


def test_read_reference_species():
    items = util.read_reference("species")
    assert len(items) >= 225936
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
        '<--results|ocr_details-->', 'ocr_results_split-->', 'image_links-->'
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

        # Test appending to existing file
        df = util.ocr_data_df(
            {
                "institutional label 3": {
                    "family":"family",
                    "id":"id",
                    "extra1":"extra3",
                },
                "institutional label 4": {
                    "family":"family",
                    "id":"id",
                    "extra2":"extra3",
                }
            },
            output_path = output_path,
        )
        assert (df.columns == required_columns + ["extra1", "extra2"]).all()
        assert len(df) == 2
        assert output_path.exists()
        output_text = output_path.read_text()
        assert "extra1" in output_text
        assert "extra3" in output_text
        for i in range(1,5):
            assert f"institutional label {i}" in output_text



def test_ocr_data_df_ocr_results():
    df = util.ocr_data_df(
        {
            "institutional label": {
                "family":"family",
                "id":"id",
                "species_ocr_results": [
                    dict(ocr="TrOCR", original_text_detected="zostericolumXX", adjusted_text="zostericolum", match_score=0.9),
                    dict(ocr="TrOCR", original_text_detected="z", adjusted_text="z", match_score=0),
                    dict(ocr="Tesseract", original_text_detected="zasdfoppasf", adjusted_text="zasdfoppasf", match_score=''),
                ],
                "extra": [],
            }
        }
    )
    required_columns = [
        'institutional label', 'id', 'family', 'genus', 'species',
       'infrasp_taxon', 'authority', 'collector_number', 'collector',
       'locality', 'geolocation', 'year', 'month', 'day',
       '<--results|ocr_details-->', 'species_ocr_results', 'image_links-->',
       'ocr_results_split-->', 'species_TrOCR_original',
       'species_TrOCR_adjusted', 'species_TrOCR_match_score',
       'species_Tesseract_original', 'species_Tesseract_adjusted',
       'species_Tesseract_match_score','extra'
    ]        
    assert len(df) == 1
    assert (df.columns == required_columns).all()
    assert (df.loc[0,'species_TrOCR_original'] == np.array(["zostericolumXX", "z"])).all()
    assert (df.loc[0,'species_TrOCR_adjusted'] == np.array(["zostericolum", "z"])).all()
    assert (df.loc[0,'species_TrOCR_match_score'] == np.array([0.9,0])).all()
    assert df.loc[0,'species_Tesseract_match_score'] == ''
    assert df.loc[0,"species_Tesseract_original"] == 'zasdfoppasf'
    assert df.loc[0,"species_Tesseract_adjusted"] == 'zasdfoppasf'
    assert df.loc[0,"extra"] == ''


def test_ocr_data_df_ocr_results(capsys):
    df = util.ocr_data_df(
        {
            "institutional label": {
                "species":"zostericolum",
                "species_image": "species.jpg",
                "id":"id",
                "species_ocr_results": [
                    dict(ocr="TrOCR", original_text_detected="zostericolumXX", adjusted_text="zostericolum", match_score=0.9),
                    dict(ocr="Tesseract", original_text_detected="zasdfoppasf", adjusted_text="zasdfoppasf", match_score=''),
                    dict(ocr="LLM", original_text_detected="zostericolum", adjusted_text="", match_score=''),
                ],
            }
        }
    )

    util.ocr_data_print_tables(df)
    captured = capsys.readouterr()
    # breakpoint()
    assert "│ species │ zostericolum │ zasdfoppasf │ zostericolumXX →       │ zostericolum │\n" in captured.out


def test_flatten_single_item_lists():
    assert util.flatten_single_item_lists(['1','2','3']) == ['1','2','3']
    assert util.flatten_single_item_lists(['1']) == '1'
    assert util.flatten_single_item_lists([]) == ''

    