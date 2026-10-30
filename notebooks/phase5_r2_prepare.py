"""R2 Phase 5 Colab preparation; safe after runtime variable loss."""
from pathlib import Path
import os
import sys
import json
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PROJECT_DIR = Path('/content/ai-image-fft-detector')
assert (PROJECT_DIR / 'config_r2.yaml').is_file(), 'Existing research checkout is missing'
sys.path.insert(0, str(PROJECT_DIR / 'src'))
sys.path.insert(0, str(PROJECT_DIR))
from aidetect.io_utils import load_config, config_hash, sha256_of_file
from aidetect.feature_extraction import read_feature_batches, _validate_existing_batch, _id_digest
from scripts.extract_features import _load_rows

R2_CONFIG_PATH = PROJECT_DIR / 'config_r2.yaml'
cfg = load_config(R2_CONFIG_PATH)
def _config_path(key):
    path = Path(cfg['paths'][key])
    return path if path.is_absolute() else PROJECT_DIR / path
DATA_ROOT, FEATURES_ROOT, REPORTS_ROOT, ARTIFACTS_ROOT = (
    _config_path(k) for k in ('data_root', 'features_root', 'reports_root', 'artifacts_root'))
FREEZE_LOCK = ARTIFACTS_ROOT / 'FREEZE.lock'
assert not FREEZE_LOCK.exists(), 'Phase 5 cannot run after freeze'
assert not list((ARTIFACTS_ROOT / 'models').glob('*.joblib')), 'Unexpected existing models'
EXPECTED = {
    'config_sha256': 'ac92656ca11539bf8aaa7c6c89d9851ddf006ed4fc514a99ba6b2e9bae24d7d6',
    'splits_sha256': '1e823b5df253951bc2a7221b618fc7c0e5716f85066d509a3d9442c56ccfef93',
    'active_manifest_sha256': '2af4d01ee69adadc0f28c741ad2bafed846d73c958b0f114efca939122ee05e8',
    'audit_result_sha256': '0ec413d2f64e3e9da9da6e97be41c5bf9071a061cf987a472bacdafaa7fce145'}
actual = {
    'config_sha256': config_hash(cfg),
    'splits_sha256': sha256_of_file(DATA_ROOT / 'splits.csv'),
    'active_manifest_sha256': sha256_of_file(DATA_ROOT / 'audit/active_manifest.csv'),
    'audit_result_sha256': sha256_of_file(DATA_ROOT / 'audit/audit_result.json')}
assert actual == EXPECTED, (actual, EXPECTED)
audit = json.loads((DATA_ROOT / 'audit/audit_result.json').read_text())
assert audit['status'] == 'passed' and audit['protocol'] == 'R2'
assert audit['config_hash'] == actual['config_sha256']
record = json.loads((ARTIFACTS_ROOT / 'splits.json').read_text())
for key in ('splits_sha256', 'active_manifest_sha256', 'audit_result_sha256'):
    assert record[key] == actual[key], key
assert record['config_hash'] == actual['config_sha256']
assert (ARTIFACTS_ROOT / 'splits.sha256').read_text().strip() == actual['splits_sha256']
frozen = pd.read_csv(DATA_ROOT / 'splits.csv', dtype={'id': str, 'group_id': str})
assert frozen.groupby('group_id')['split'].nunique().max() == 1
counts = frozen.groupby(['split', 'label']).size().to_dict()
assert counts == {('train',0):1750, ('train',1):1750, ('val',0):375, ('val',1):375, ('test',0):375, ('test',1):375}
for split in ('train', 'val', 'test'):
    assert len(_load_rows(DATA_ROOT, split, expected_active_digest=actual['active_manifest_sha256'])) == int((frozen['split'] == split).sum())
P5_BACKUP = DATA_ROOT.parent / 'review_backup' / ('phase_5_extraction_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
P5_BACKUP.mkdir(parents=True, exist_ok=False)
cli_env = os.environ.copy()
cli_env['PYTHONPATH'] = str(PROJECT_DIR / 'src') + os.pathsep + str(PROJECT_DIR)
P5_RESULTS = {}
print('R2 Phase 5 prerequisites VERIFIED', json.dumps(actual, indent=2))
print('Frozen class counts:', counts)
print('Batch size:', cfg['batch_size'], 'Feature root:', FEATURES_ROOT)
print('Evidence directory:', P5_BACKUP)
