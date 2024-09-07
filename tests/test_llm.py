import pytest
from pathlib import Path
from unittest.mock import patch

from hespi.llm import llm_correct_detection_results, build_template, encode_image, output_parser, get_llm

test_data_dir = Path(__file__).parent/"testdata"

def mock_llm(*args, **kwargs):
    def invoke(*args, **kwargs):
        return f"""
        family: Chlorophyceae
        genus: Chlamydomonas
        species: reinhardtii
        garbage: garbage
        """
    return invoke

def test_encode_image():
    image = test_data_dir/"test.jpg"
    encoded_image = encode_image(image)
    assert encoded_image is not None
    assert encoded_image.startswith("/9j/4AAQSkZJRgABAQAASABIAAD/4QB")


def test_build_template():
    institutional_label_image = test_data_dir/"test.jpg"
    detection_results = {
        "family": "Piurnosacedie",
        "authority": "Ffisell .",
        "species": "hilli",
        "year": "1986",
        "collector": "Pips Lyndon",
        "locality": "Vieforca\nAeonpatha",
        "month": "March",
    }
    template = build_template(institutional_label_image, detection_results)
    template_string = template.invoke({}).to_string()
    assert template_string is not None
    assert template_string.startswith("System: You are an expert")
    assert "ata:image/jpeg;base64,/9j/4AAQSkZJRg" in template_string
    assert "\nAI: Certainly, here are the corrections:\n" in template_string
    assert "family: Piurnosacedie" in template_string
    assert "authority: Ffisell ." in template_string
    assert "species: hilli" in template_string
    assert "year: 1986" in template_string
    assert "collector: Pips Lyndon" in template_string
    assert "locality: Vieforca Aeonpatha" in template_string


def test_output_parser():
    text = """
    family: Piurnosacedie
    authority: Ffisell .
    species: hilli
    year: 1986
    collector: Pips Lyndon
    locality: Vieforca Aeonpatha
    month: March
    these are the corrected fields.
    ----
    """
    result = output_parser(text)
    assert result == {
        "family": "Piurnosacedie",
        "authority": "Ffisell .",
        "species": "hilli",
        "year": "1986",
        "collector": "Pips Lyndon",
        "locality": "Vieforca Aeonpatha",
        "month": "March",
    }


def test_get_llm_error():
    with pytest.raises(ValueError):
        get_llm("unknown_model")


@patch("hespi.llm.ChatOpenAI", mock_llm)
def test_llm():
    institutional_label_image = test_data_dir/"institution_label.jpg"
    detection_results = {
        "id": "MELUD104449_sp66541195778794889279_medium",
        "label_classification": "handwritten",
        "predictions": Path(
            "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.all.jpg"
        ),
        "collector_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.collector.jpg"
            )
        ],
        "collector_ocr_results": [
            {
                "ocr": "Tesseract",
                "original_text_detected": "MO. HAAN",
                "adjusted_text": "MO. HAAN",
                "match_score": "",
            }
        ],
        "species_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.species.jpg"
            )
        ],
        "collector_number_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.collector_number.jpg"
            )
        ],
        "authority_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.authority.jpg"
            )
        ],
        "year_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.year.jpg"
            )
        ],
        "year_ocr_results": [
            {
                "ocr": "Tesseract",
                "original_text_detected": "Moy",
                "adjusted_text": "Moy",
                "match_score": "",
            }
        ],
        "month_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.month.jpg"
            )
        ],
        "locality_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.locality.jpg"
            )
        ],
        "genus_image": [
            Path(
                "hespi-output/MELUD104449_sp66541195778794889279_medium/MELUD104449_sp66541195778794889279_medium.institutional_label/MELUD104449_sp66541195778794889279_medium.institutional_label.genus.jpg"
            )
        ],
        "collector": "MO. HAAN",
        "year": "Moy",
    }
    llm_correct_detection_results(mock_llm(), institutional_label_image, detection_results)
    assert detection_results["family"] == "Chlorophyceae"
    assert detection_results["genus"] == "Chlamydomonas"
    assert detection_results["species"] == "reinhardtii"
    assert detection_results['family_ocr_results'][0] == {'ocr': 'LLM', 'original_text_detected': 'Chlorophyceae', 'adjusted_text': '', 'match_score': 0}
