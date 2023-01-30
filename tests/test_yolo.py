import tempfile
from pathlib import Path
import shutil
import torch

from hespi import yolo


def test_yolo_output():
    base_dir = Path(__file__).parent/"testdata"
    images = [
        base_dir/"test.jpg",
        base_dir/"test2.jpg",
    ]

    class MockYoloOutput():
        def __init__(self):
            self.names = [f"category{i}" for i in range(4)]
            self.pred = torch.tensor([
                [[0,0,10,10,-1,0],[0,10,10,20,-1,1]],
                [[10,10,20,20,-1,2],[10,0,20,10,-1,3]],
            ])
        
        def save(self, save_dir:Path, **kwargs):
            for image in images:
                shutil.copy(image, save_dir/image.name )


    class MockYoloModel():
        def predict(self, images):
            return MockYoloOutput()

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