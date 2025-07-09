import tempfile
from pathlib import Path
from hespi.report import write_report
from hespi import util

def test_write_report():
    with tempfile.TemporaryDirectory() as tmpdir:   
        report_file = Path(tmpdir) / "report.html"
        component_files = {'stub': [Path('stub.primary_specimen_label.jpg'), Path('stub.swatch.jpg')]}
        ocr_df = util.ocr_data_df(
            {
                "primary specimen label": {
                    "species_image":"species_image.jpg",
                    "species_image_2":"species_image_2.jpg",
                    "id": "stub",
                    "species_ocr_results": [
                        dict(ocr="TrOCR", original_text_detected="zostericolumXX", adjusted_text="zostericolum", match_score=0.9),
                        dict(ocr="Tesseract", original_text_detected="zasdfoppasf", adjusted_text="zasdfoppasf", match_score=''),
                    ],
                    "extra": [],
                }
            }
        )

        write_report(report_file, component_files=component_files, ocr_df=ocr_df)

        assert report_file.exists()
        text = report_file.read_text()
        assert "<head>" in text
        assert "species_image.jpg" in text
        assert "species_image_2.jpg" in text