from pathlib import Path
import numpy as np

from analysis.detection import detect_poster_elements


def test_end_to_end_detection_monkeypatch(monkeypatch, tmp_path):
    weights_path = tmp_path / "dummy-best.pt"
    weights_path.write_bytes(b"dummy")

    # create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # monkeypatch PosterDetector.predict_array to return a fake detection
    def fake_predict_array(self, array, conf=0.25):
        h, w = array.shape[:2]
        return [
            {"class_id": 3, "class_name": "Product", "confidence": 0.9, "x1": int(0.1*w), "y1": int(0.1*h), "x2": int(0.5*w), "y2": int(0.6*h)}
        ]

    monkeypatch.setattr("models.detector.PosterDetector.predict_array", fake_predict_array)

    records, source = detect_poster_elements(img, image_path=None, weights_path=weights_path)
    assert source == "model"
    assert len(records) == 1
    rec = records[0]
    assert rec.class_name == "Product"
