# Portable project bundles

V0.8 bundles a completed trained project into an inference-only `.tsppe.zip` archive. The bundle contains a portable `project.json`, `bundle_manifest.json`, `weights/person_best.pt`, `weights/ppe_best.pt`, and an explicitly referenced compliance policy when present. It excludes raw/prepared datasets, training runs, caches, logs, and source media.

## Export, verify, import

```bash
two-stage-ppe export --project runs/site/project.json --output site.tsppe.zip
two-stage-ppe verify site.tsppe.zip
two-stage-ppe inspect site.tsppe.zip
two-stage-ppe import site.tsppe.zip --output projects/site
```

The imported `project.json` has relative model paths and can be passed directly to `PPEPipeline.from_project()`. Export does not change the source project.

## Integrity and security

`bundle_manifest.json` uses schema version 1 and inventories every bundled payload file with its portable path, size, and SHA256 digest. Verification checks the manifest, supported schema, project status/configuration, exact inventory, model references, non-empty weights, hashes, duplicate paths, unsafe paths, and symlink-like ZIP entries. Import always verifies before extraction and refuses an existing destination.

Integrity hashes detect accidental modification or corruption; they do not provide publisher identity or cryptographic signing. Archives are untrusted input: no archive content is executed or deserialized during verification, and extraction is performed entry-by-entry only after path validation.

## Portability and compatibility

Export rewrites bundled model references to `weights/person_best.pt` and `weights/ppe_best.pt`, removes source dataset/output configuration paths, and rejects remaining absolute paths in the portable manifest. This makes imported projects independent of the exporting machine's root path. Existing project manifests remain loadable; bundle export requires a completed project with both non-empty model files.
