import tempfile
from pathlib import Path
import shutil
import torch
from unittest.mock import patch

from hespi import yolo
    
class MockBoxes():
    def __init__(self, boxes):
        self.xyxy = torch.tensor([boxes[0:4]])
        self.cls = torch.tensor(boxes[-1])


class MockImageResult():
    def __init__(self, boxes):
        self.boxes = [MockBoxes(box) for box in boxes]


class MockYoloOutput():
    def __init__(self, images):
        self.images = images
        self.names = [f"category{i}" for i in range(4)]
        self.predictions = [
            MockImageResult([[0,0,10,10,-1,0],[0,10,10,20,-1,1]]),
            MockImageResult([[10,10,20,20,-1,2],[10,0,20,10,-1,3]]),
        ]
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index < len(self.predictions):
            p = self.predictions[self._current_index]
            self._current_index += 1
            return p

        raise StopIteration


class MockYoloPredictor():
    save_dir = ""

    def __init__(self, *args):
        pass

    def setup_model(self, model):
        pass


class MockYoloModel():
    predictor = None
    model = None
    names = [str(x) for x in range(4)]
    
    def predict(self, source, show=False, save=True, batch=4, imgsz=1280):
        for index, image in enumerate(source):
            shutil.copy(image, self.predictor.save_dir/image.name )

        return MockYoloOutput(source)

    def to(self, device):
        self.device = device


@patch('hespi.yolo.DetectionPredictor', return_value=MockYoloPredictor())
def test_yolo_output(mock_detection_predictor):
    base_dir = Path(__file__).parent/"testdata"
    images = [
        base_dir/"test.jpg",
        base_dir/"test2.jpg",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:        
        files = yolo.yolo_output(
            MockYoloModel(),
            images,
            tmpdir,
        )

        assert len(files) == len(images)
        assert files["test"][0].exists()
        assert files["test"][1].exists()
        assert files["test2"][0].exists()
        assert files["test2"][1].exists()
        assert files["test"][0].parent.name == "test"
        assert files["test2"][0].parent.name == "test2"
        assert files["test"][0].parent.parent == Path(tmpdir)        