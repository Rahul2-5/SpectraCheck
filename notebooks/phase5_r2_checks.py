"""Verify readable features and sealed provenance; save Phase 5 evidence."""
import pyarrow.parquet as pq
checks = {}
for split in ('train', 'val', 'test'):
    rows = _load_rows(DATA_ROOT, split, expected_active_digest=actual['active_manifest_sha256']).sort_values('id')
    folder = FEATURES_ROOT / ('test_sealed' if split == 'test' else split)
    paths = sorted(folder.glob('batch_*.parquet'))
    assert len(paths) == (len(rows) + cfg['batch_size'] - 1) // cfg['batch_size']
    total = 0
    for index, path in enumerate(paths):
        batch = rows.iloc[index * cfg['batch_size']:(index + 1) * cfg['batch_size']]
        _validate_existing_batch(path, cfg_hash=actual['config_sha256'], splits_hash=actual['splits_sha256'], split=split, batch_index=index, batch_ids_hash=_id_digest(batch['id']))
        metadata = pq.read_metadata(path)
        assert metadata.num_rows == len(batch)
        total += metadata.num_rows
    checks[split] = {'rows': total, 'batches': len(paths), 'metadata_matches': True}
    if split != 'test':
        frame = read_feature_batches(FEATURES_ROOT, split, artifacts_root=ARTIFACTS_ROOT)
        assert set(frame['id']) == set(rows['id']) and len(frame) == len(rows)
        expected_rows = rows.set_index('id')
        for column in ('label', 'group_id'):
            assert frame.set_index('id')[column].sort_index().equals(expected_rows[column].sort_index())
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        assert sum(c.startswith('pixel_') for c in frame.columns) == 1024
        assert sum(c.startswith('rapsd_ring_') for c in frame.columns) == 64
        checks[split]['all_numeric_finite'] = True
        checks[split]['pixel_features'] = 1024
        checks[split]['rapsd_rings'] = 64
        del frame
try:
    read_feature_batches(FEATURES_ROOT, 'test', artifacts_root=ARTIFACTS_ROOT)
except PermissionError as error:
    print('Expected sealed-test denial:', error)
else:
    raise AssertionError('Test reader did not enforce seal')
test_run = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_pixel_baseline.py', 'tests/test_feature_extraction.py', 'tests/test_r2.py'], cwd=PROJECT_DIR, env=cli_env, capture_output=True, text=True)
(P5_BACKUP / 'pytest.txt').write_text(test_run.stdout + test_run.stderr)
print('Focused Colab tests:', test_run.returncode, test_run.stdout, test_run.stderr)
test_run.check_returncode()
assert not FREEZE_LOCK.exists()
assert sha256_of_file(DATA_ROOT / 'splits.csv') == actual['splits_sha256']
evidence = {'status': 'completed', 'phase': 5, 'protocol': 'R2', **actual, 'checks': checks, 'extraction': P5_RESULTS, 'test_feature_values_read': False, 'sealed_reader_denied': True, 'training_started': False, 'freeze_created': False, 'evaluation_started': False, 'supervisor_signoff': 'unverified', 'original_raw_audit': 'blocked', 'audit_domain': 'treated_input', 'pytest_output': test_run.stdout, 'backup': str(P5_BACKUP)}
report = P5_BACKUP / 'phase_5_r2_checks.json'
report.write_text(json.dumps(evidence, indent=2, sort_keys=True))
manifest = {p.name: sha256_of_file(p) for p in P5_BACKUP.iterdir() if p.is_file()}
(P5_BACKUP / 'sha256.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
print('R2 PHASE 5 COMPLETE', json.dumps(evidence, indent=2, sort_keys=True))
print('Review record SHA256:', sha256_of_file(report))
print('STOP before Phase 6. No training, freeze or evaluation ran.')
