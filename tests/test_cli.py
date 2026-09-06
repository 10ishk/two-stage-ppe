import pytest

from two_stage_ppe.cli import build_parser, main


def test_cli_parses_detect_options():
    args = build_parser().parse_args(["detect", "--input", "in", "--output", "out", "--person-model", "p.pt", "--ppe-model", "c.pt", "--person-class", "3", "--ppe-batch-size", "8", "--save-json"])
    assert args.command == "detect"
    assert args.person_class == "3"
    assert args.crop_padding == 0.05
    assert args.save_json
    assert args.ppe_batch_size == 8


@pytest.mark.parametrize("value", ["0", "-3"])
def test_cli_rejects_non_positive_batch_size(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["detect", "--input", "in", "--output", "out", "--person-model", "p.pt", "--ppe-model", "c.pt", "--ppe-batch-size", value])


def test_cli_rejects_no_outputs():
    with pytest.raises(SystemExit, match="Nothing to save"):
        main(["detect", "--input", "in", "--output", "out", "--person-model", "p.pt", "--ppe-model", "c.pt", "--no-images"])
