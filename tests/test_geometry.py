import pytest

from two_stage_ppe.geometry import clip_box, crop_to_global, expand_box, ioa, iou


def test_padding_expands_each_side_by_fraction():
    assert expand_box((10, 20, 30, 60), 0.1, 100, 100) == (8, 16, 32, 64)


def test_padding_clips_at_image_boundaries():
    assert expand_box((0, 0, 20, 20), 0.5, 25, 25) == (0, 0, 25, 25)


def test_clipping():
    assert clip_box((-4, 5, 120, 80), 100, 60) == (0, 5, 100, 60)


def test_required_crop_to_global_transform():
    assert crop_to_global((10, 20, 60, 100), (100, 50)) == (110, 70, 160, 150)


def test_iou_and_ioa():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert ioa((5, 0, 10, 10), (0, 0, 20, 20)) == 1.0


def test_degenerate_ioa_is_zero():
    assert ioa((1, 1, 1, 3), (0, 0, 5, 5)) == 0.0

