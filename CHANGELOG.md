# Changelog

All notable changes are documented here.

## 0.8.0

- Add portable, inference-only trained-project ZIP bundles.
- Add export, inspect, verify, and safe import CLI/API workflows.
- Add SHA256 inventory verification, archive path validation, and cross-machine relative model paths.
- Support explicitly referenced compliance policy files in bundles.

## 0.7.0

- Add configurable PPE compliance policies with structured observed/missing results.
- Add YAML/JSON policy loading to image and video CLI workflows.
- Add optional JSON/JSONL compliance fields, OpenCV compliance labels, and frame-observation totals.
- Preserve tracking IDs while evaluating compliance independently for every processed frame.

## 0.6.0

- Add explicit stage-aware training resume and idempotent completed-project reuse.
- Verify prepared datasets, checkpoints, validation metadata, and calibration artifacts.
- Add dependency-scoped configuration and practical source-dataset fingerprints.
- Add read-only dry-run resume plans and controlled `--restart-from` invalidation.
- Preserve failure context while enabling successful recovery on later runs.
- Evolve project manifests to schema v2 while retaining v0.5 inference loading.

## 0.5.0

- Add end-to-end two-stage Ultralytics training orchestration.
- Add Pascal VOC dataset auditing and leakage-safe parent/child preparation.
- Organize person and PPE best checkpoints with validation metrics and failure status.
- Add structured training results, project manifests, dry runs, and prepare-only mode.
- Add optional validation-only confidence calibration and project-based inference loading.

## 0.4.0

- Add optional persistent worker tracking with an isolated ByteTrack adapter.
- Add `track_id` to tracked video person results while retaining frame-local `id` semantics.
- Render persistent IDs on person labels and stream them through JSONL output.
- Reset default tracker state between video calls and expose lightweight track counts.
- Preserve backward-compatible non-tracked image, directory, video, and PPE result behavior.

## 0.3.0

- Add incremental video inference without loading complete videos into memory.
- Add annotated OpenCV video output and streaming JSONL frame predictions.
- Add configurable frame stride plus start-frame and processed-frame limits.
- Reuse existing batched PPE crop inference independently for each processed frame.

## 0.2.0

- Batch PPE crops from each source image in one ordered detector call by default.
- Add configurable PPE crop chunking for accelerator-memory control.
- Preserve stable crop-to-person ownership when invalid crops are skipped.
- Add architectural regression coverage for backend call counts, chunking, and result-count validation.

## 0.1.0

- Introduce the two-stage person-to-PPE Python API and CLI.
- Add global coordinate remapping, hierarchical JSON results, and OpenCV rendering.
- Add reusable dataset preparation, cascade evaluation, and threshold-calibration tools.
