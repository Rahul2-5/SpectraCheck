"""R2 Phase 6 readiness; run after the verified Phase 5 checks."""
assert evidence['status'] == 'completed' and evidence['phase'] == 5
assert evidence['sealed_reader_denied'] and not evidence['test_feature_values_read']
assert config_hash(cfg) == EXPECTED['config_sha256']
assert sha256_of_file(DATA_ROOT / 'splits.csv') == EXPECTED['splits_sha256']
assert not FREEZE_LOCK.exists()
assert not (ARTIFACTS_ROOT / 'EVALUATION_STARTED.json').exists()
assert not (ARTIFACTS_ROOT / 'training_results.json').exists(), 'Existing results require review before any rerun'
assert not list((ARTIFACTS_ROOT / 'models').glob('*.joblib')), 'Existing models require review'
assert cfg['cv_folds'] == 5 and cfg['search_iters'] == 30 and cfg['seed'] == 42
assert cfg['training']['search_n_jobs'] == 1
P6_BACKUP = DATA_ROOT.parent / 'review_backup' / ('phase_6_training_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
P6_BACKUP.mkdir(parents=True, exist_ok=False)
P6_START = {'status': 'ready', 'config_hash': config_hash(cfg), 'splits_sha256': EXPECTED['splits_sha256'], 'phase5_record_sha256': sha256_of_file(report), 'test_features_read': False, 'backup': str(P6_BACKUP)}
(P6_BACKUP / 'start.json').write_text(json.dumps(P6_START, indent=2, sort_keys=True))
print('R2 PHASE 6 READY', json.dumps(P6_START, indent=2))
print('Six matched CPU pipelines; five shared folds; 30 search trials per model. Test stays sealed.')
