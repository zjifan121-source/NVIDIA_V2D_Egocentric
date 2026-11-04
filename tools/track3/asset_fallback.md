# Input-reviewed asset fallback

Keep an existing complete reconstruction. If its original initialization failed
an input-only review of an asymmetric structure's direction, allow one fixed
alternative mesh with newly estimated matching poses. The alternative must pass
the same review and cover every frame. Other failures remain unresolved.

This entry selects and copies existing bundles. It does not generate meshes,
estimate poses, run GPU inference, score against ground truth, or change the
default reconstruction pipeline. Structural review is explicit human input;
this is not an automatic visual classifier. Freeze that review and the candidate
before evaluation. A lower depth residual alone is not a structural pass.

## Run

Requires Python 3.11+ and NumPy.

```sh
python tools/track3/asset_fallback.py --manifest experiment/manifest.json
python tools/track3/asset_fallback.py --manifest experiment/manifest.json --output delivery/new_run
python -m unittest discover -s tools/track3/tests -p test_asset_fallback.py -v
```

Without `--output`, validate all selected bundles and print the selection.
With it, copy to a new directory, verify each copied hash and write `run.json`
and `selection.json`. Existing output directories are refused. Incomplete cases
exit nonzero and cannot produce a partial delivery.

## Manifest

All paths may be relative to the manifest directory. The following is a schema
example; replace placeholders with hashes from the frozen producer receipts.

```json
{
  "schema_version": 1,
  "cases": [{
    "id": "example_sequence",
    "frames": 249,
    "objects": ["white_pot"],
    "baseline": {
      "state": "FAIL_ANCHOR_ALIGNMENT",
      "input_signature": "OBSERVATIONS_AND_FP_CONFIG_HASH",
      "gate": {
        "state": "FAIL",
        "reason": "ASYMMETRIC_STRUCTURE_DIRECTION",
        "GT_used": false,
        "evidence": [{"path": "old_anchor_review.json", "sha256": "REVIEW_HASH"}]
      }
    },
    "candidate": {
      "state": "COMPLETE",
      "input_signature": "OBSERVATIONS_AND_FP_CONFIG_HASH",
      "gate": {
        "state": "PASS",
        "GT_used": false,
        "evidence": [{"path": "new_anchor_review.json", "sha256": "REVIEW_HASH"}]
      },
      "bundle": {
        "path": "new_bundle",
        "files": {"result.npz": "RESULT_HASH", "mesh.obj": "MESH_HASH"}
      },
      "binding": {
        "convention": "T_world_from_camera @ T_camera_from_original_metric_mesh",
        "bundle_sha256": "FILES_DICTIONARY_HASH",
        "evidence": [{"path": "mesh_pose_receipt.json", "sha256": "RECEIPT_HASH"}]
      }
    }
  }]
}
```

For a complete baseline, use `state: COMPLETE` with the same `bundle`/`binding`
fields as the candidate. It is retained regardless of candidate scores; no
candidate is required. Include all referenced textures and materials in `files`.
Object order matches `object_names`; meshes are `mesh.obj`, `mesh_1.obj`, etc.

`bundle_sha256` is SHA-256 of UTF-8
`json.dumps(files, sort_keys=True, separators=(",", ":"))`.
The producer's binding receipt must record which metric mesh generated which
poses, including axes/origin, already-baked scale, and public FP pose convention.
The validator checks receipt and bundle integrity, frame coverage, finite poses,
object identity/order and `world = camera @ object_to_camera`. It cannot establish
correct physical correspondence from an incorrect producer attestation. Never
create a new binding to disguise an old-pose/new-mesh combination.

The observation signature must cover RGB, K, metric depth, mask, frame order,
anchor, FP settings and runtime version, excluding only the changed mesh and
appearance. Verify these source hashes before constructing the manifest. A
matching string alone is not an independent input audit. Retain the original
review receipts, overlays and inference receipts outside the delivery directory.

## Evidence and limits

On the development Public20 cohort, one input-gate-triggered replacement recovered
ep44's complete 249-frame reconstruction (local AUC 0.983685). Keeping the other
19 bundles fixed raised the operational mean from 0.797960 to 0.847144. This
cohort includes earlier assisted identity inputs; it is not a fresh automatic
holdout. Only one case triggered this fallback, so generalization is unproven.

Replacing the mesh in all nine pot episodes was rejected: among eight complete
matched pairs, mean scene AUC decreased by 0.108361. Do not use the scores to
choose per-episode assets after evaluation. These are Public reconstruction
metrics, not policy tracking or hidden-test scores.
