"""Approved R2 treated-input audit; raw failed evidence is never overwritten."""
from io import BytesIO
import json
from pathlib import Path
import shutil

import imagehash
import pandas as pd
from PIL import Image

from aidetect.audit import audit_table, duplicate_groups, estimate_jpeg_quality, shortcut_probe
from aidetect.data import _atomic_json
from aidetect.io_utils import config_hash, sha256_of_file
from aidetect.preprocess import R2_PREPROCESS_VERSION, _canonical_patch


def run_r2_audit(cfg: dict) -> dict:
    """Copy verified original cohort into a new root, measure one common treatment."""
    policy = cfg["r2"]
    if (cfg["preprocess"] != R2_PREPROCESS_VERSION or not policy.get("approval")
        or policy["audit_domain"] != "treated_input" or cfg["audit"]["shortcut_max_accuracy"] != .60):
        raise ValueError("R2 requires its approved treatment and unchanged threshold")
    root = Path(cfg["paths"]["data_root"]).resolve()
    source = Path(policy["source_data_root"]).resolve()
    artifacts = Path(cfg["paths"]["artifacts_root"])
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("R2 and original experiment roots must be separate")
    old_artifacts = Path(policy.get("source_artifacts_root", "artifacts"))
    if any(p.exists() for p in (source / "splits.csv", old_artifacts / "FREEZE.lock",
                                old_artifacts / "EVALUATION_STARTED.json")):
        raise ValueError("Original experiment is sealed; R2 cannot reuse it")
    if any(p.exists() for p in (root / "splits.csv", artifacts / "splits.sha256",
                                artifacts / "FREEZE.lock", artifacts / "EVALUATION_STARTED.json")):
        raise ValueError("R2 splits/freeze already exist; do not rerun the audit")
    raw_path = source / "audit/audit_result.json"
    if sha256_of_file(raw_path) != policy["source_audit_sha256"]:
        raise ValueError("Approved raw audit evidence changed")
    raw = json.loads(raw_path.read_text())
    if raw["status"] != "blocked" or raw["target_jpeg_quality"] != cfg["preprocess"]["jpeg_quality"]:
        raise ValueError("Raw failed audit/quality does not match R2 decision")
    source_manifest = source / "main/manifest.csv"
    if sha256_of_file(source_manifest) != raw["specification"]["source_manifest_sha256"]:
        raise ValueError("Original manifest changed")
    source_active = source / "audit/active_manifest.csv"
    if sha256_of_file(source_active) != raw["active_manifest_sha256"]:
        raise ValueError("Original audited cohort changed")
    ids = pd.read_csv(source_active, dtype={"id": str})[["id", "label"]]
    original = pd.read_csv(source_manifest, dtype={"id": str})
    if ids.id.duplicated().any() or original.id.duplicated().any():
        raise ValueError("Duplicate provenance IDs")
    selected = ids.merge(original, on="id", suffixes=("_active", ""), how="left", validate="one_to_one")
    if selected.sha256.isna().any() or not selected.label.eq(selected.label_active).all():
        raise ValueError("Cohort IDs/labels do not match raw acquisition")
    counts = selected.label.value_counts()
    if set(counts.index) != {0, 1} or counts[0] != counts[1] or counts.min() < cfg["dataset"]["minimum_ai_images"]:
        raise ValueError("R2 needs the balanced, sufficiently large approved cohort")
    digest = config_hash(cfg)
    output = root / "audit"
    result_path = output / "audit_result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("status") != "incomplete" or previous.get("config_hash") != digest:
            raise ValueError("Existing R2 audit must be preserved; not overwritten")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Unmanaged R2 root; inspect before starting")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, {"status": "incomplete", "config_hash": digest})
    shutil.copy2(raw_path, output / "raw_prior_audit.json")
    # Re-measure originals; the old audit's CSV is evidence, not trusted input
    # for R2 grouping or metadata. audit_table also verifies source bytes.
    print("R2 independently measuring approved original cohort", flush=True)
    before = audit_table(selected.drop(columns="label_active"), source / "main",
                         phash_size=cfg["audit"]["phash_size"])
    rows = []
    for index, row in enumerate(before.to_dict("records")):
        src = (source / "main" / row["rel_path"]).resolve()
        dst = (root / "main" / row["rel_path"]).resolve()
        if not src.is_relative_to(source / "main") or not dst.is_relative_to(root / "main"):
            raise ValueError("Source path escapes collection")
        if sha256_of_file(src) != row["sha256"]:
            raise ValueError(f"Original source bytes changed: {row['id']}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        if sha256_of_file(dst) != row["sha256"]:
            raise ValueError("R2 original-byte copy changed; no silent repair")
        patch = _canonical_patch(dst, cfg)
        from hashlib import sha256
        with Image.open(BytesIO(patch)) as image:
            quality, residual = estimate_jpeg_quality(image.quantization)
            rows.append({**row, "rel_path": "main/" + row["rel_path"],
                "source_width": row["width"], "source_height": row["height"],
                "source_n_bytes": row["n_bytes"], "source_format": row["format"],
                "source_sha256": row["sha256"], "original_sha256": row["sha256"],
                "canonical_sha256": sha256(patch).hexdigest(),
                "original_phash": row["phash"], "phash": str(imagehash.phash(image, hash_size=cfg["audit"]["phash_size"])),
                "width": image.width, "height": image.height, "aspect_ratio": 1.,
                "format": "JPEG", "mode": "RGB", "is_jpeg": 1, "n_bytes": len(patch),
                "jpeg_quality": quality, "jpeg_quality_table_mae": residual,
                "config_hash": digest, "metadata_domain": "treated_input",
                "source_config_hash": row["config_hash"]})
        if (index + 1) % cfg["batch_size"] == 0:
            print(f"R2 verified and measured {index + 1}/{len(before)}", flush=True)
    active = pd.DataFrame(rows)
    active["group_id"] = duplicate_groups(active, cfg["audit"]["duplicate_hamming_distance"])
    options = dict(cv_folds=cfg["cv_folds"], tree_depth=cfg["audit"]["probe_tree_depth"],
                   logistic_max_iter=cfg["audit"]["probe_logistic_max_iter"])
    raw_probe = shortcut_probe(before, cfg["seed"], **options)
    probe = shortcut_probe(active, cfg["seed"], **options)
    mixed = int((active.groupby("group_id").label.nunique() > 1).sum())
    before.to_csv(output / "original_audit.csv", index=False)
    active.to_csv(output / "active_manifest.csv", index=False)
    selected.drop(columns="label_active").to_csv(root / "main/manifest.csv", index=False)
    result = {"status": "passed" if probe["max_accuracy"] <= .60 and not mixed else "blocked",
        "audit_domain": "treated_input", "protocol": "R2", "config_hash": digest,
        "threshold": .60, "before_probe": raw_probe, "final_probe": probe,
        "raw_audit_status": "blocked", "raw_audit_sha256": policy["source_audit_sha256"],
        "active_manifest_sha256": sha256_of_file(output / "active_manifest.csv"),
        "original_rows": len(before), "balanced_rows": len(active), "mixed_label_groups": mixed,
        "normalization_applied": True, "target_jpeg_quality": cfg["preprocess"]["jpeg_quality"],
        "approval": policy["approval"], "supervisor_signoff": policy["supervisor_signoff"],
        "limitations": "Equal treated geometry is imposed; raw geometry, compression history and content confounding remain limitations. Amendment selected after metadata diagnostics."}
    _atomic_json(root / "main/acquisition.json", {"status": "complete", "kind": "verified_local_R2_copy",
        "config_hash": digest, "manifest_sha256": sha256_of_file(root / "main/manifest.csv"),
        "source_manifest_sha256": raw["specification"]["source_manifest_sha256"]})
    _atomic_json(result_path, result)
    report = Path(cfg["paths"]["reports_root"])
    report.mkdir(parents=True, exist_ok=True)
    (report / "audit_report.md").write_text("# R2 treated-input audit\n\nOriginal audit remains blocked.\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    print(json.dumps(result, indent=2), flush=True)
    return result
