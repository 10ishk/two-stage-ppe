# Changelog

All notable changes are documented here.

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
