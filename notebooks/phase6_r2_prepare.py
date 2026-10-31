"""R2 Phase 6 readiness, also usable after a stopped runtime is replaced."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

PROJECT_DIR = Path('/content/ai-image-fft-detector')
sys.path.insert(0, str(PROJECT_DIR / 'src'))
from aidetect.io_utils import config_hash, load_config, sha256_of_file

R2_CONFIG_PATH = PROJECT_DIR / 'config_r2.yaml'
cfg = load_config(R2_CONFIG_PATH)
def _phase6_path(key):
    path = Path(cfg['paths'][key])
    return path if path.is_absolute() else PROJECT_DIR / path
DATA_ROOT, FEATURES_ROOT, REPORTS_ROOT, ARTIFACTS_ROOT = (
    _phase6_path(key) for key in ('data_root', 'features_root', 'reports_root', 'artifacts_root'))
FREEZE_LOCK = ARTIFACTS_ROOT / 'FREEZE.lock'
summary = json.loads((PROJECT_DIR / 'reports/phase_5_r2_checks.json').read_text())
assert summary['status'] == 'completed', 'Reviewed Phase 5 completion is required'
report = Path(summary['drive_backup']) / 'phase_5_r2_checks.json'
assert sha256_of_file(report) == summary['review_record_sha256'], 'Phase 5 record changed'
evidence = json.loads(report.read_text())
assert evidence['status'] == 'completed' and evidence['phase'] == 5
assert evidence['sealed_reader_denied'] and not evidence['test_feature_values_read']
assert config_hash(cfg) == evidence['config_sha256']
assert sha256_of_file(DATA_ROOT / 'splits.csv') == evidence['splits_sha256']
for filename, key in (('active_manifest.csv', 'active_manifest_sha256'),
                      ('audit_result.json', 'audit_result_sha256')):
    assert sha256_of_file(DATA_ROOT / 'audit' / filename) == evidence[key]
EXPECTED = {'config_sha256': evidence['config_sha256'], 'splits_sha256': evidence['splits_sha256']}
assert not FREEZE_LOCK.exists()
assert not (ARTIFACTS_ROOT / 'EVALUATION_STARTED.json').exists()
if 'p6_process' in globals():
    assert p6_process.poll() is not None, 'The previous training subprocess is still active'
P6_CHECKPOINT_ROOT = DATA_ROOT.parent / 'phase6_checkpoints'
assert not (P6_CHECKPOINT_ROOT / 'running.lock').exists(), 'Checkpoint writer is active or left a stale lock; inspect before resuming'
if not (P6_CHECKPOINT_ROOT / 'manifest.json').exists():
    assert not (ARTIFACTS_ROOT / 'training_results.json').exists(), 'Existing results require review before any rerun'
    assert not list((ARTIFACTS_ROOT / 'models').glob('*.joblib')), 'Existing models require review'
# The CLI verifies checkpoint ownership, input/code/version hashes and final outputs before reuse.
assert cfg['cv_folds'] == 5 and cfg['search_iters'] == 30 and cfg['seed'] == 42
assert cfg['training']['search_n_jobs'] == 1
cli_env = os.environ.copy()
cli_env['PYTHONPATH'] = str(PROJECT_DIR / 'src') + os.pathsep + str(PROJECT_DIR)
P6_BACKUP = DATA_ROOT.parent / 'review_backup' / ('phase_6_training_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
P6_BACKUP.mkdir(parents=True, exist_ok=False)
P6_START = {'status': 'ready', 'config_hash': config_hash(cfg), 'splits_sha256': EXPECTED['splits_sha256'], 'phase5_record_sha256': sha256_of_file(report), 'test_features_read': False, 'backup': str(P6_BACKUP), 'checkpoint_dir': str(P6_CHECKPOINT_ROOT)}
(P6_BACKUP / 'start.json').write_text(json.dumps(P6_START, indent=2, sort_keys=True))
print('R2 PHASE 6 READY', json.dumps(P6_START, indent=2))
print('Six matched CPU pipelines; five shared folds; 30 trials per model. Checkpoints on Drive; test stays sealed.')
