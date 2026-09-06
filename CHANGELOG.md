# Changelog

All notable changes are documented here.

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
