"""Run the explicitly labelled supervisor development demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aidetect.io_utils import load_config
from app.backend.demo import DevelopmentDemoService
from app.backend.main import create_app


MODEL_PATH = "artifacts/r2_fast_rf_t4/models/frequency_svm.joblib"
MANIFEST_PATH = ROOT / "reports/phase_6_fast_rf_t4_backup_hashes.json"


def build_demo_app():
    settings = load_config(ROOT / "app/backend/settings.yaml")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    settings.update(
        demo_mode=True,
        demo_model_path=MODEL_PATH,
        demo_model_sha256=manifest["artifacts/models/frequency_svm.joblib"],
    )
    service = DevelopmentDemoService(ROOT, settings)
    return create_app(ROOT, service=service, settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the labelled SpectraCheck development demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(build_demo_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
