import warnings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from enum import Enum
from pathlib import Path
import pytesseract


class TrOCRSize(Enum):
    SMALL = "small"
    BASE = "base"
    LARGE = "large"

    def __str__(self):
        return str(self.value)


class OCR:
    def get_text(self, image_path: Path) -> str:
        raise NotImplementedError


class TrOCR(OCR):
    def __init__(self, size: TrOCRSize = TrOCRSize.BASE):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            print("Getting TrOCRProcessor")
            self.processor = TrOCRProcessor.from_pretrained(
                f"microsoft/trocr-{size}-handwritten"
            )

            print("Getting VisionEncoderDecoderModel")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                f"microsoft/trocr-{size}-handwritten"
            )

    def get_text(self, image_path: Path) -> str:
        image = Image.open(image_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pixel_values = self.processor(
                images=image, return_tensors="pt"
            ).pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return generated_text


class Tesseract(OCR):
    def __init__(self):
        super().__init__()
        
        self.no_tesseract = False # This will be set to True later if it isn't found.

    def get_text(self, image_path: Path) -> str:
        if self.no_tesseract:
            return None

        try:
            return pytesseract.image_to_string(str(image_path)).strip()
        except Exception as err:
            print(f"No tesseract available: {err}")
            self.no_tesseract = True
        
        return None

