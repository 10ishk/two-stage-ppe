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

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [type("Output", (), {"boxes": Boxes()})()]


def test_adapter_normalizes_mock_backend_output():
    model = Model()
    adapter = UltralyticsDetector(model)
    detections = adapter.predict(np.zeros((20, 20, 3)), conf=0.3, iou=0.45, device="cpu")
    assert detections[0].class_name == "custom-item"
    assert detections[0].bbox == (1.0, 2.0, 10.0, 20.0)
    assert model.kwargs["verbose"] is False
    assert model.kwargs["device"] == "cpu"


def test_adapter_handles_empty_outputs():
    model = Model()
    model.predict = lambda **kwargs: []
    assert UltralyticsDetector(model).predict(np.zeros((1, 1, 3)), conf=0.3, iou=0.4, device=None) == []

