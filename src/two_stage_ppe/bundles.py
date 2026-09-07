"""Portable, integrity-checked inference project bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from importlib.metadata import PackageNotFoundError, version


BUNDLE_SCHEMA_VERSION = 1
_MAX_MANIFEST_BYTES = 1_000_000
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


def _framework_version() -> str:
    try:
        return version("two-stage-ppe")
    except PackageNotFoundError:
        return "0.0.0"


@dataclass(frozen=True)
class BundleInspection:
    kind: str
    valid: bool
    schema_version: int | None
    framework_version: str | None
    project_status: str | None
    person_model: str | None
    ppe_model: str | None
    parent_class: str | None
    child_classes: list[str]
    inference: dict[str, Any]
    files: list[dict[str, Any]]
    policy: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_bundle_path(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def _project_manifest_path(project: str | Path) -> Path:
    path = Path(project)
    if path.is_dir():
        path = path / "project.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_project(project: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _project_manifest_path(project)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid project manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Project manifest must contain an object")
    if payload.get("status") not in {"completed", "complete"}:
        raise ValueError(f"Project is not completed: {payload.get('status', 'unknown')}")
    models = payload.get("models")
    if not isinstance(models, dict) or not models.get("person") or not models.get("ppe"):
        raise ValueError("Project manifest must contain person and PPE model paths")
    return path, payload


def _is_absolute_text(value: str) -> bool:
    return value.startswith("/") or bool(_WINDOWS_ABSOLUTE.match(value))


def _portable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    portable = copy.deepcopy(payload)
    portable["models"] = {"person": "weights/person_best.pt", "ppe": "weights/ppe_best.pt"}
    portable["portable"] = True
    configuration = portable.get("configuration")
    if isinstance(configuration, dict):
        configuration.pop("dataset", None)
        configuration.pop("output", None)

    def reject_absolute(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                reject_absolute(child)
        elif isinstance(value, list):
            for child in value:
                reject_absolute(child)
        elif isinstance(value, str) and _is_absolute_text(value):
            raise ValueError("Project manifest contains an absolute path that cannot be bundled portably")

    reject_absolute(portable)
    return portable


def _model_source(root: Path, value: str) -> Path:
    source = Path(value)
    return source if source.is_absolute() else root / source


def _policy_source(root: Path, payload: dict[str, Any]) -> tuple[Path, str] | None:
    reference = payload.get("policy_path")
    if reference is None and isinstance(payload.get("compliance_policy"), dict):
        reference = payload["compliance_policy"].get("path")
    if reference is None:
        return None
    if not isinstance(reference, str) or _is_absolute_text(reference):
        raise ValueError("Referenced compliance policy must use a relative path")
    source = root / reference
    if not source.is_file():
        raise FileNotFoundError(f"Referenced compliance policy is missing: {source}")
    return source, f"policies/{source.name}"


def _entry(path: str, data: bytes) -> dict[str, Any]:
    return {"path": path, "size": len(data), "sha256": _sha256(data)}


def _zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data)


def export_project(project: str | Path, output: str | Path) -> Path:
    """Create an inference-only portable bundle without changing the source project."""
    manifest_path, project_payload = _read_project(project)
    root = manifest_path.parent
    portable = _portable_payload(project_payload)
    files: dict[str, bytes] = {"project.json": (json.dumps(portable, indent=2, sort_keys=True) + "\n").encode("utf-8")}
    for role, bundled_name in (("person", "weights/person_best.pt"), ("ppe", "weights/ppe_best.pt")):
        source = _model_source(root, str(project_payload["models"][role]))
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Required {role} model is missing or empty: {source}")
        files[bundled_name] = source.read_bytes()
    policy = _policy_source(root, project_payload)
    if policy is not None:
        source, bundled_name = policy
        files[bundled_name] = source.read_bytes()
        portable["policy_path"] = bundled_name
        files["project.json"] = (json.dumps(portable, indent=2, sort_keys=True) + "\n").encode("utf-8")
    entries = [_entry(name, data) for name, data in sorted(files.items())]
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "framework_version": _framework_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_name": root.name,
        "project": {"status": portable.get("status"), "models": portable["models"], "classes": portable.get("classes", {}), "inference": portable.get("inference", {})},
        "files": entries,
    }
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for name, data in sorted(files.items()):
            _zip_write(archive, name, data)
        _zip_write(archive, "bundle_manifest.json", (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    verify_bundle(destination)
    return destination


def _load_bundle(path: Path) -> tuple[zipfile.ZipFile, dict[str, Any], dict[str, zipfile.ZipInfo]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid bundle ZIP: {path}") from exc
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        archive.close()
        raise ValueError("Bundle contains duplicate archive paths")
    for info in infos:
        _safe_bundle_path(info.filename)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            archive.close()
            raise ValueError(f"Bundle contains unsupported symlink entry: {info.filename}")
    mapping = {item.filename: item for item in infos}
    if "bundle_manifest.json" not in mapping or "project.json" not in mapping:
        archive.close()
        raise ValueError("Bundle must contain bundle_manifest.json and project.json")
    if mapping["bundle_manifest.json"].file_size > _MAX_MANIFEST_BYTES:
        archive.close()
        raise ValueError("Bundle manifest is too large")
    try:
        payload = json.loads(archive.read("bundle_manifest.json").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        archive.close()
        raise ValueError("Bundle manifest is malformed") from exc
    if not isinstance(payload, dict):
        archive.close()
        raise ValueError("Bundle manifest must contain an object")
    return archive, payload, mapping


def verify_bundle(bundle: str | Path) -> BundleInspection:
    """Verify untrusted ZIP structure and SHA256 integrity without loading models."""
    path = Path(bundle)
    archive, payload, infos = _load_bundle(path)
    try:
        if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported bundle schema version")
        entries = payload.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Bundle manifest has no file inventory")
        inventory: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("size"), int) or not isinstance(entry.get("sha256"), str):
                raise ValueError("Bundle manifest contains an invalid file entry")
            name = _safe_bundle_path(entry["path"]).as_posix()
            if name in inventory:
                raise ValueError("Bundle manifest contains duplicate file entries")
            inventory[name] = entry
        expected = set(inventory) | {"bundle_manifest.json"}
        if set(infos) != expected:
            raise ValueError("Bundle archive files do not match its manifest inventory")
        for name, entry in inventory.items():
            data = archive.read(name)
            if len(data) != entry["size"] or _sha256(data) != entry["sha256"]:
                raise ValueError(f"Integrity check failed for {name}")
        try:
            project = json.loads(archive.read("project.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Bundled project.json is malformed") from exc
        if not isinstance(project, dict) or project.get("status") not in {"completed", "complete"}:
            raise ValueError("Bundled project is not completed")
        models = project.get("models", {})
        for role, expected_name in (("person", "weights/person_best.pt"), ("ppe", "weights/ppe_best.pt")):
            value = models.get(role)
            if value != expected_name or value not in inventory:
                raise ValueError(f"Bundled project has an invalid {role} model reference")
            if inventory[value]["size"] <= 0:
                raise ValueError(f"Bundled {role} model is empty")
        if any(_is_absolute_text(value) for value in _strings(project)):
            raise ValueError("Bundled project contains an absolute path")
        classes = project.get("classes", {})
        parent = next(iter(classes.get("parent", {}).values()), None) if isinstance(classes, dict) else None
        children = list((classes.get("children", {}) or {}).values()) if isinstance(classes, dict) else []
        policy = {"path": project["policy_path"]} if isinstance(project.get("policy_path"), str) else None
        if policy and project["policy_path"] not in inventory:
            raise ValueError("Bundled policy reference is missing")
        return BundleInspection("bundle", True, payload["schema_version"], payload.get("framework_version"), project.get("status"), models.get("person"), models.get("ppe"), parent, children, project.get("inference", {}), [inventory[name] for name in sorted(inventory)], policy)
    finally:
        archive.close()


def _strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def inspect_project(source: str | Path) -> BundleInspection:
    path = Path(source)
    if path.suffix.lower() == ".zip":
        return verify_bundle(path)
    manifest, payload = _read_project(path)
    models = payload["models"]
    classes = payload.get("classes", {})
    parent = next(iter(classes.get("parent", {}).values()), None) if isinstance(classes, dict) else None
    children = list((classes.get("children", {}) or {}).values()) if isinstance(classes, dict) else []
    files = []
    for role in ("person", "ppe"):
        candidate = _model_source(manifest.parent, models[role])
        files.append({"path": str(models[role]), "size": candidate.stat().st_size if candidate.is_file() else None, "exists": candidate.is_file()})
    return BundleInspection("project", all(item["exists"] and item["size"] for item in files), payload.get("schema_version"), None, payload.get("status"), models.get("person"), models.get("ppe"), parent, children, payload.get("inference", {}), files, {"path": payload["policy_path"]} if payload.get("policy_path") else None)


def import_project(bundle: str | Path, output: str | Path) -> Path:
    """Verify then safely extract a portable project for normal relative-path loading."""
    inspection = verify_bundle(bundle)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    archive, _payload, infos = _load_bundle(Path(bundle))
    try:
        destination.mkdir(parents=True)
        root = destination.resolve()
        for name in sorted(infos):
            if name == "bundle_manifest.json":
                continue
            target = (destination / _safe_bundle_path(name)).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe extraction target: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    except Exception:
        # The newly created directory contains only this failed import's content.
        import shutil
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        archive.close()
    # Recheck extracted content against the verified inventory before returning it.
    for entry in inspection.files:
        path = destination / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"] or _sha256(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Extracted bundle integrity check failed for {entry['path']}")
    return destination / "project.json"


def format_inspection(inspection: BundleInspection) -> str:
    lines = [f"Kind: {inspection.kind}", f"Valid: {inspection.valid}", f"Project status: {inspection.project_status}", f"Person model: {inspection.person_model}", f"PPE model: {inspection.ppe_model}", f"Parent class: {inspection.parent_class}", f"PPE classes: {', '.join(inspection.child_classes)}", f"Inference: {json.dumps(inspection.inference, sort_keys=True)}"]
    if inspection.framework_version:
        lines.insert(2, f"Framework version: {inspection.framework_version}")
    if inspection.schema_version is not None:
        lines.insert(2, f"Schema version: {inspection.schema_version}")
    if inspection.policy:
        lines.append(f"Compliance policy: {inspection.policy['path']}")
    lines.extend(f"File: {item['path']} ({item.get('size')} bytes)" for item in inspection.files)
    return "\n".join(lines)
