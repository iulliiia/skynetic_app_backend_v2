"""Tests for the inference service. Run: venv/bin/python -m pytest -q

The fast tests use a fake predictor. The real-model test runs only when
AI_REAL_MODEL=1 and model2.h5 is present (it takes ~10 s to load)."""
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


class FakePredictor:
    def __init__(self, scores=(0.2, 0.7, 0.1)):
        self.scores = scores
        self.batches = []

    def predict(self, batch):
        self.batches.append(batch)
        return np.array([self.scores], dtype=np.float32)


class BrokenPredictor:
    def predict(self, batch):
        raise RuntimeError("boom")


def jpeg_bytes(size=(300, 240), mode="RGB", fmt="JPEG"):
    img = Image.new(mode, size, color=(120, 80, 60) if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def fake():
    return FakePredictor()


@pytest.fixture
def client(fake):
    with TestClient(main.create_app(lambda: fake, token="secret")) as c:
        yield c


def post(client, data, token="secret", filename="x.jpg"):
    headers = {main.TOKEN_HEADER: token} if token is not None else {}
    return client.post("/v1/predict", files={"image": (filename, data, "image/jpeg")}, headers=headers)


def test_health_reports_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model"]["class_labels"] == ["low_risk", "high_risk", "clear_skin"]
    assert body["model"]["input_size"] == [180, 180]


def test_predict_returns_structured_result(client, fake):
    r = post(client, jpeg_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] == 1
    assert body["predicted_label"] == "high_risk"
    assert body["predicted_name_ro"] == "Grad de risc crescut"
    assert body["scores"] == pytest.approx([0.2, 0.7, 0.1])
    assert body["model"] == {"name": "skynetic-cnn", "version": main.MODEL_VERSION, "framework": main.FRAMEWORK}
    assert body["inference_ms"] >= 0
    # the model saw exactly one 180x180 RGB image with raw 0-255 values
    batch = fake.batches[0]
    assert batch.shape == (1, 180, 180, 3) and batch.dtype == np.float32
    assert 0 <= batch.min() and batch.max() <= 255 and batch.max() > 1


def test_preprocess_matches_training_pipeline():
    # grayscale PNG and RGBA PNG both become RGB 180x180
    for mode in ("L", "RGBA"):
        batch = main.preprocess(jpeg_bytes(size=(640, 480), mode=mode, fmt="PNG"))
        assert batch.shape == (1, 180, 180, 3)
    # pixel values are not rescaled or contrast-enhanced
    flat = main.preprocess(jpeg_bytes(size=(200, 200), fmt="PNG"))
    assert flat[0, 90, 90].tolist() == pytest.approx([120, 80, 60], abs=1)


def test_token_required(client):
    assert post(client, jpeg_bytes(), token="wrong").status_code == 401
    assert post(client, jpeg_bytes(), token=None).status_code == 401
    assert post(client, jpeg_bytes(), token="wrong").json()["error"]["code"] == "unauthorized"


def test_token_optional_when_not_configured(fake):
    with TestClient(main.create_app(lambda: fake, token="")) as c:
        assert post(c, jpeg_bytes(), token=None).status_code == 200


def test_rejects_invalid_images(client):
    r = post(client, b"GIF89a not really an image")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    truncated = jpeg_bytes()[: len(jpeg_bytes()) // 2]
    assert post(client, truncated).status_code == 400
    assert post(client, b"").status_code == 400


def test_rejects_oversized_upload(client):
    r = post(client, b"\xff" * (main.MAX_BYTES + 1))
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_missing_file_field(client):
    r = client.post("/v1/predict", headers={main.TOKEN_HEADER: "secret"}, files={"other": ("x.jpg", b"123", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_failed"


def test_inference_failure_is_reported():
    with TestClient(main.create_app(BrokenPredictor, token="secret")) as c:
        r = post(c, jpeg_bytes())
        assert r.status_code == 500 and r.json()["error"]["code"] == "inference_failed"


def test_model_load_failure_degrades_health():
    def boom():
        raise RuntimeError("no model file")

    with TestClient(main.create_app(boom, token="secret")) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model_loaded"] is False
        r = post(c, jpeg_bytes())
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


@pytest.mark.skipif(os.environ.get("AI_REAL_MODEL") != "1" or not os.path.exists(main.MODEL_PATH),
                    reason="set AI_REAL_MODEL=1 with model2.h5 present")
def test_real_model_is_deterministic_on_sample_images():
    samples = [os.path.join("..", "skynetic_app", "assets", "images", f"scan{i}.jpg") for i in (1, 2, 3)]
    with TestClient(main.create_app(token="")) as c:
        assert c.get("/health").json()["model_loaded"] is True
        for path in samples:
            data = open(path, "rb").read()
            a = post(c, data, token=None).json()
            b = post(c, data, token=None).json()
            assert a["scores"] == b["scores"], "same input must give the same scores"
            assert sum(a["scores"]) == pytest.approx(1.0, abs=1e-3), "softmax output"
            assert a["predicted_class"] in (0, 1, 2)
