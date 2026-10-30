"""Exploratory fixed-crop metadata diagnostic; never authorizes research splits.

Images are transformed in memory only. The original audit remains authoritative.
This candidate is deliberately separate from aidetect.preprocess_image.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from aidetect.audit import estimate_jpeg_quality, shortcut_probe
from aidetect.io_utils import config_hash, load_config, sha256_of_file


def crop_metadata(payload: bytes, size: int, quality: int) -> dict:
    """Measure one identical RGB centre crop/JPEG treatment, without resizing."""
    if size < 1 or not 1 <= quality <= 100:
        raise ValueError("Invalid crop size or JPEG quality")
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb.size
        if min(width, height) < size:
            raise ValueError("Candidate rejects undersized images; no upscaling")
        left, top = (width - size) // 2, (height - size) // 2
        patch = rgb.crop((left, top, left + size, top + size))
        stream = BytesIO()
        patch.save(stream, format="JPEG", quality=quality, subsampling=0,
                   optimize=False, progressive=False)
    encoded = stream.getvalue()
    with Image.open(BytesIO(encoded)) as measured:
        estimated, residual = estimate_jpeg_quality(measured.quantization)
        return {"is_jpeg": 1, "jpeg_quality": estimated,
                "jpeg_quality_table_mae": residual, "width": measured.width,
                "height": measured.height, "n_bytes": len(encoded),
                "aspect_ratio": measured.width / measured.height,
                "crop_sha256": sha256(encoded).hexdigest()}


def run_candidate(cfg: dict, project: Path, output: Path) -> dict:
    """Measure the same approved-audit population; persist diagnostic records only."""
    project, output = Path(project).resolve(), Path(output).resolve()
    def resolve_setting(key):
        path = Path(cfg["paths"][key])
        return (path if path.is_absolute() else project / path).resolve()
    root, artifacts = resolve_setting("data_root"), resolve_setting("artifacts_root")
    protected = [root, artifacts, resolve_setting("features_root")]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("Output must be outside data, artifacts and feature roots")
    if output.exists():
        raise ValueError("Use a fresh output directory; prior diagnostics are preserved")
    if any(p.exists() for p in (root / "splits.csv", artifacts / "splits.sha256",
                                artifacts / "FREEZE.lock", artifacts / "EVALUATION_STARTED.json")):
        raise ValueError("Research is sealed or split; candidate diagnostic is prohibited")
    audit_path, active_path = root / "audit/audit_result.json", root / "audit/active_manifest.csv"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "blocked" or audit["config_hash"] != config_hash(cfg):
        raise ValueError("Requires a blocked audit for the unchanged current config")
    if sha256_of_file(active_path) != audit["active_manifest_sha256"]:
        raise ValueError("Active audit manifest changed")
    manifest_path = root / "main/manifest.csv"
    if sha256_of_file(manifest_path) != audit["specification"]["source_manifest_sha256"]:
        raise ValueError("Original acquisition manifest changed")
    active = pd.read_csv(active_path, dtype={"id": str})
    original = pd.read_csv(manifest_path, dtype={"id": str})
    if active.id.duplicated().any() or original.id.duplicated().any():
        raise ValueError("Duplicate source IDs")
    joined = active[["id", "label"]].merge(original, on="id", how="left",
        suffixes=("_active", ""), validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or not joined.label.eq(joined.label_active).all():
        raise ValueError("Active IDs/labels disagree with original acquisition")
    quality = audit.get("target_jpeg_quality")
    if not isinstance(quality, int):
        raise ValueError("No recorded median JPEG quality; do not invent one")
    output.mkdir(parents=True)
    rows = []
    for index, row in enumerate(joined.to_dict("records")):
        source = (root / "main" / row["rel_path"]).resolve()
        if not source.is_relative_to(root / "main"):
            raise ValueError("Source escapes the original collection")
        payload = source.read_bytes()
        if sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"Original image changed: {row['id']}")
        rows.append({"id": row["id"], "label": int(row["label"]),
                     "source_sha256": row["sha256"],
                     **crop_metadata(payload, cfg["image_size"], quality)})
        if (index + 1) % cfg["batch_size"] == 0:
            print(f"Measured {index + 1}/{len(joined)} in-memory candidate crops", flush=True)
    frame = pd.DataFrame(rows)
    probe = shortcut_probe(frame, cfg["seed"], cv_folds=cfg["cv_folds"],
        tree_depth=cfg["audit"]["probe_tree_depth"],
        logistic_max_iter=cfg["audit"]["probe_logistic_max_iter"])
    frame.to_csv(output / "candidate_metadata.csv", index=False)
    result = {"status": "exploratory_diagnostic_only", "research_audit_status": "blocked",
        "config_hash": config_hash(cfg), "crop_size": cfg["image_size"],
        "jpeg_quality": quality, "seed": cfg["seed"], "rows": len(frame),
        "threshold": cfg["audit"]["shortcut_max_accuracy"], "candidate_probe": probe,
        "candidate_below_threshold": probe["max_accuracy"] <= cfg["audit"]["shortcut_max_accuracy"],
        "source_audit_sha256": sha256_of_file(audit_path),
        "source_active_manifest_sha256": sha256_of_file(active_path),
        "research_splits_created": False, "derived_images_persisted": False,
        "limitation": "Constant treated geometry is designed, not evidence that original geometry, compression history or content confounding was removed. Review required even if probes are below threshold."}
    (output / "candidate_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_candidate(load_config(args.config), PROJECT, args.output_dir)


if __name__ == "__main__":
    main()
