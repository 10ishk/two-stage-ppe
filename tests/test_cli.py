import pytest

from two_stage_ppe.cli import build_parser, main


def test_cli_parses_detect_options():
    args = build_parser().parse_args(["detect", "--input", "in", "--output", "out", "--person-model", "p.pt", "--ppe-model", "c.pt", "--person-class", "3", "--ppe-batch-size", "8", "--save-json", "--policy", "site.yaml"])
    assert args.command == "detect"
    assert args.person_class == "3"
    assert args.crop_padding == 0.05
    assert args.save_json
    assert args.ppe_batch_size == 8
    assert args.policy == "site.yaml"


@pytest.mark.parametrize("value", ["0", "-3"])
def test_cli_rejects_non_positive_batch_size(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["detect", "--input", "in", "--output", "out", "--person-model", "p.pt", "--ppe-model", "c.pt", "--ppe-batch-size", value])


def test_cli_rejects_no_outputs():
    with pytest.raises(SystemExit, match="Nothing to save"):
        main(["detect", "--input", "in", "--output", "out", "--person-model", "p.pt", "--ppe-model", "c.pt", "--no-images"])


def test_video_cli_parses_sampling_and_jsonl_options():
    args = build_parser().parse_args([
        "video", "--input", "input.mp4", "--output", "output.mp4",
        "--person-model", "p.pt", "--ppe-model", "c.pt",
        "--frame-stride", "2", "--start-frame", "3", "--max-frames", "5",
        "--ppe-batch-size", "8", "--jsonl", "results.jsonl", "--no-render", "--track", "--policy", "site.json",
    ])
    assert args.command == "video"
    assert (args.frame_stride, args.start_frame, args.max_frames) == (2, 3, 5)
    assert args.ppe_batch_size == 8
    assert args.jsonl == "results.jsonl"
    assert args.no_render
    assert args.track
    assert args.policy == "site.json"


@pytest.mark.parametrize(("option", "value"), [("--frame-stride", "0"), ("--start-frame", "-1"), ("--max-frames", "0")])
def test_video_cli_rejects_invalid_controls(option, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "video", "--input", "input.mp4", "--person-model", "p.pt", "--ppe-model", "c.pt",
            option, value,
        ])


def test_python_module_video_help(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "two_stage_ppe", "video", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--frame-stride" in completed.stdout
    assert "--track" in completed.stdout
