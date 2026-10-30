"""Deterministic test extraction only; never inspect feature values."""
assert set(P5_RESULTS) >= {'train', 'val'}, 'Run train and validation extraction first'
print('Writing R2 test features to test_sealed; values remain sealed.', flush=True)
result = subprocess.run([sys.executable, '-u', 'scripts/extract_features.py', '--config', str(R2_CONFIG_PATH), '--split', 'test'], cwd=PROJECT_DIR, env=cli_env, capture_output=True, text=True)
(P5_BACKUP / 'test_extraction.txt').write_text(result.stdout + result.stderr)
print(result.stdout, result.stderr, flush=True)
result.check_returncode()
P5_RESULTS['test'] = json.loads(result.stdout)
