import numpy as np

import two_stage_ppe.visualization as visualization
from two_stage_ppe.results import Detection, ImageResult, PersonResult


def test_renderer_shows_track_id_only_on_person_label(monkeypatch):
    labels = []
    monkeypatch.setattr(
        visualization,
        "_draw_label",
        lambda image, text, origin, color: labels.append(text),
    )
    result = ImageResult(
        "frame.jpg",
        100,
        100,
        [
            PersonResult(
                0,
                (10, 10, 80, 90),
                0.91,
                [Detection(0, "helmet", (20, 15, 40, 35), 0.88)],
                track_id=17,
            )
        ],
    )
    visualization.render_result(np.zeros((100, 100, 3), dtype=np.uint8), result)
    assert labels == ["person #17 0.91", "helmet 0.88"]


def test_renderer_is_unchanged_without_tracking(monkeypatch):
    labels = []
    monkeypatch.setattr(
        visualization,
        "_draw_label",
        lambda image, text, origin, color: labels.append(text),
    )
    result = ImageResult("image.jpg", 20, 20, [PersonResult(0, (1, 1, 10, 15), 0.7)])
    visualization.render_result(np.zeros((20, 20, 3), dtype=np.uint8), result)
    assert labels == ["person 0.70"]

