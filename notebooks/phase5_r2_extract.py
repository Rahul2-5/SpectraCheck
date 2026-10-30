"""Run only train and validation extraction, preserving exact CLI outputs."""
for split in ('train', 'val'):
    print('Extracting R2', split, flush=True)
    result = subprocess.run([sys.executable, '-u', 'scripts/extract_features.py', '--config', str(R2_CONFIG_PATH), '--split', split], cwd=PROJECT_DIR, env=cli_env, capture_output=True, text=True)
    (P5_BACKUP / (split + '_extraction.txt')).write_text(result.stdout + result.stderr)
    print(result.stdout, result.stderr, flush=True)
    result.check_returncode()
    P5_RESULTS[split] = json.loads(result.stdout)
