# PPE compliance policies

Compliance policies interpret model observations against a site-defined list of required PPE classes. They do not establish real-world safety compliance: results depend on the configured classes, detector observations, and inference thresholds.

## Policy file

YAML and JSON policy files contain one active policy:

```yaml
name: construction-default
required:
  - hard-hat
  - vest
  - boots
min_confidence: 0.30
unknown_class_behavior: unknown
```

`name` is required. `required` must be a non-empty, duplicate-free list of exact PPE model class names. `min_confidence` is optional and filters only detections that already passed normal PPE inference thresholds. `unknown_class_behavior` is either `unknown` (the default) or `error`. With `unknown`, a requirement unavailable in the PPE model produces `unknown`; with `error`, inference fails before processing starts. No fuzzy class-name matching or default policy exists.

## Python API

```python
from two_stage_ppe import CompliancePolicy, PPEPipeline

policy = CompliancePolicy(name="site-default", required=["hard-hat", "vest", "boots"])
result = PPEPipeline("person.pt", "ppe.pt").predict("image.jpg", compliance_policy=policy)

# Existing model-free results can be evaluated separately.
policy.evaluate_image(result, model_classes=["hard-hat", "vest", "boots"])
```

Each evaluated person receives an optional `compliance` object. All requirements observed means `compliant`; otherwise it is `non_compliant` and lists `missing`. A policy unavailable in the supplied PPE model is `unknown` unless strict error mode is selected. Extra PPE and duplicate detections do not reduce compliance.

## CLI and video

```bash
two-stage-ppe detect --input image.jpg --output output --person-model person.pt --ppe-model ppe.pt --policy examples/compliance_policy.yaml
two-stage-ppe video --input input.mp4 --jsonl output.jsonl --person-model person.pt --ppe-model ppe.pt --policy examples/compliance_policy.yaml --track
```

Policy evaluation occurs independently on every processed video frame. `track_id` remains a persistent identity when tracking is enabled, but the status is not smoothed, stored as worker history, or treated as a permanent condition. Video summaries count frame observations, not unique workers.
