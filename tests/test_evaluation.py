from evaluate_cascade import Box, evaluate_records, match_boxes


def test_duplicate_prediction_is_false_positive():
    truth = [Box((0, 0, 10, 10), 1)]
    predictions = [Box((0, 0, 10, 10), 1, 0.9), Box((0, 0, 10, 10), 1, 0.8)]
    assert match_boxes(truth, predictions) == (1, 1, 0)


def test_matching_is_class_aware():
    assert match_boxes([Box((0, 0, 10, 10), 0)], [Box((0, 0, 10, 10), 1)]) == (0, 1, 1)


def test_hierarchical_evaluation():
    truth = {"images": [{"image": "x.jpg", "parents": [{"bbox": [0, 0, 20, 20]}], "children": [{"bbox": [5, 5, 10, 10], "class_id": 2}]}]}
    prediction = {"images": [{"image": "x.jpg", "persons": [{"bbox": [0, 0, 20, 20], "confidence": 0.8, "ppe": [{"bbox": [5, 5, 10, 10], "class_id": 2, "confidence": 0.7}]}]}]}
    result = evaluate_records(truth, prediction)
    assert result["parent"]["f1"] == 1.0
    assert result["child"]["f1"] == 1.0

