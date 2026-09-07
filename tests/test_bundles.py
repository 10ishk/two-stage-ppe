import json
import zipfile
from pathlib import Path

import pytest

from two_stage_ppe import PPEPipeline
from two_stage_ppe.bundles import export_project, import_project, inspect_project, verify_bundle
from two_stage_ppe.cli import build_parser


def make_project(root: Path, *, policy=False) -> Path:
    (root / "weights").mkdir(parents=True)
    (root / "weights/person_best.pt").write_bytes(b"person-weight")
    (root / "weights/ppe_best.pt").write_bytes(b"ppe-weight")
    payload = {
        "schema_version": 2, "status": "completed",
        "models": {"person": "weights/person_best.pt", "ppe": "weights/ppe_best.pt"},
        "classes": {"parent": {"0": "person"}, "children": {"0": "helmet", "1": "vest"}},
        "inference": {"person_conf": 0.25, "ppe_conf": 0.3, "crop_padding": 0.05, "iou": 0.45},
        "configuration": {"dataset": "removed-on-export", "output": "removed-on-export"},
    }
    if policy:
        (root / "policies").mkdir()
        (root / "policies/site.yaml").write_text("name: site\nrequired: [helmet]\n", encoding="utf-8")
        payload["policy_path"] = "policies/site.yaml"
    path = root / "project.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_export_verify_inspect_and_portable_paths(tmp_path):
    source = make_project(tmp_path / "machine_a" / "run", policy=True)
    original = source.read_bytes()
    bundle = export_project(source, tmp_path / "share" / "site.tsppe.zip")
    assert source.read_bytes() == original
    inspection = verify_bundle(bundle)
    assert inspection.valid and inspection.person_model == "weights/person_best.pt"
    assert inspection.policy == {"path": "policies/site.yaml"}
    with zipfile.ZipFile(bundle) as archive:
        project = json.loads(archive.read("project.json"))
        assert project["models"] == {"person": "weights/person_best.pt", "ppe": "weights/ppe_best.pt"}
        assert "removed-on-export" not in archive.read("project.json").decode()
        assert "policies/site.yaml" in archive.namelist()
    assert inspect_project(bundle).kind == "bundle"
    assert inspect_project(source).kind == "project"


def test_import_is_cross_root_safe_and_loadable(tmp_path):
    bundle = export_project(make_project(tmp_path / "machine_a" / "run"), tmp_path / "share.tsppe.zip")
    imported = import_project(bundle, tmp_path / "other_machine" / "projects" / "site")
    assert imported == tmp_path / "other_machine" / "projects" / "site" / "project.json"

    class Detector:
        class_names = {0: "person"}

    pipeline = PPEPipeline.from_project(imported, _person_detector=Detector(), _ppe_detector=Detector())
    assert pipeline.config.person_conf == 0.25
    assert imported.read_text(encoding="utf-8").find(str(tmp_path / "machine_a")) == -1


def test_hash_corruption_missing_models_and_existing_target_fail(tmp_path):
    project = make_project(tmp_path / "run")
    bundle = export_project(project, tmp_path / "valid.zip")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(bundle, "a") as archive:
            archive.writestr("weights/person_best.pt", b"modified")
    with pytest.raises(ValueError, match="duplicate|Integrity"):
        verify_bundle(bundle)
    (project.parent / "weights/ppe_best.pt").unlink()
    with pytest.raises(FileNotFoundError):
        export_project(project, tmp_path / "missing.zip")
    good = export_project(make_project(tmp_path / "other"), tmp_path / "good.zip")
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(FileExistsError):
        import_project(good, target)


@pytest.mark.parametrize("name", ["../evil.txt", "/absolute.txt", "nested/../../evil.txt"])
def test_unsafe_archive_paths_are_rejected(tmp_path, name):
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(name, b"x")
        archive.writestr("project.json", b"{}")
        archive.writestr("bundle_manifest.json", json.dumps({"schema_version": 1, "files": []}))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        verify_bundle(bundle)


def test_duplicate_and_malformed_bundle_manifest_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("project.json", b"{}")
            archive.writestr("project.json", b"{}")
            archive.writestr("bundle_manifest.json", b"{}")
    with pytest.raises(ValueError, match="duplicate"):
        verify_bundle(duplicate)
    malformed = tmp_path / "malformed.zip"
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("project.json", b"{}")
        archive.writestr("bundle_manifest.json", b"not-json")
    with pytest.raises(ValueError, match="malformed"):
        verify_bundle(malformed)


def test_bundle_cli_parsing():
    parser = build_parser()
    assert parser.parse_args(["export", "--project", "run", "--output", "out.tsppe.zip"]).command == "export"
    assert parser.parse_args(["inspect", "bundle.tsppe.zip"]).command == "inspect"
    assert parser.parse_args(["verify", "bundle.tsppe.zip"]).command == "verify"
    assert parser.parse_args(["import", "bundle.tsppe.zip", "--output", "run"]).command == "import"
