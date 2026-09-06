import numpy as np

from two_stage_ppe.detectors import UltralyticsDetector


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Boxes:
    xyxy = Tensor([[1, 2, 10, 20]])
    conf = Tensor([0.75])
    cls = Tensor([1])

    def __len__(self):
        return 1


class Model:
    names = {0: "helmet", 1: "custom-item"}

    def __init__(self):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return [type("Output", (), {"boxes": Boxes()})() for _ in kwargs["source"]]


def test_adapter_normalizes_mock_backend_output():
    model = Model()
    adapter = UltralyticsDetector(model)
    detections = adapter.predict(np.zeros((20, 20, 3)), conf=0.3, iou=0.45, device="cpu")
    assert detections[0].class_name == "custom-item"
    assert detections[0].bbox == (1.0, 2.0, 10.0, 20.0)
    assert model.kwargs["verbose"] is False
    assert model.kwargs["device"] == "cpu"
    assert isinstance(model.kwargs["source"], list)


def test_adapter_handles_empty_outputs():
    model = Model()
    model.predict = lambda **kwargs: [type("Output", (), {"boxes": None})()]
    assert UltralyticsDetector(model).predict(np.zeros((1, 1, 3)), conf=0.3, iou=0.4, device=None) == []


def test_empty_batch_skips_backend_call():
    model = Model()
    adapter = UltralyticsDetector(model)
    assert adapter.predict_batch([], conf=0.3, iou=0.4, device=None) == []
    assert model.calls == 0


def test_multiple_inputs_use_one_ordered_backend_call():
    model = Model()
    adapter = UltralyticsDetector(model)
    results = adapter.predict_batch(
        [np.zeros((2, 2, 3)), np.zeros((3, 3, 3))], conf=0.3, iou=0.4, device=None
    )
    assert model.calls == 1
    assert len(results) == 2
    assert all(result[0].class_name == "custom-item" for result in results)


def test_batch_result_count_mismatch_is_clear():
    model = Model()
    model.predict = lambda **kwargs: [type("Output", (), {"boxes": None})()]
    adapter = UltralyticsDetector(model)
    try:
        adapter.predict_batch(
            [np.zeros((2, 2, 3)), np.zeros((3, 3, 3))], conf=0.3, iou=0.4, device=None
        )
    except RuntimeError as exc:
        assert "1 result collections for 2 input images" in str(exc)
    else:
        raise AssertionError("Expected result-count validation")
