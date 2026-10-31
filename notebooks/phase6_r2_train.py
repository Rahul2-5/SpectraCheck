"""Run authorized R2 training; preserve outputs on Drive when it completes."""
import shutil
assert P6_START['status'] == 'ready'
assert not FREEZE_LOCK.exists()
P6_COMMAND = [sys.executable, '-u', 'scripts/train_models.py', '--config', str(R2_CONFIG_PATH), '--checkpoint-dir', str(P6_CHECKPOINT_ROOT)]
with (P6_BACKUP / 'training.log').open('x') as p6_log:
    p6_process = subprocess.Popen(P6_COMMAND, cwd=PROJECT_DIR, env=cli_env, stdout=p6_log, stderr=subprocess.STDOUT, text=True)
    P6_START.update(status='running', pid=p6_process.pid, command=P6_COMMAND, started_utc=datetime.now(timezone.utc).isoformat())
    (P6_BACKUP / 'start.json').write_text(json.dumps(P6_START, indent=2, sort_keys=True))
    print('R2 PHASE 6 STARTED', json.dumps(P6_START, indent=2), flush=True)
    p6_exit = p6_process.wait()
print('Training exit:', p6_exit, flush=True)
P6_START['exit_code'] = p6_exit
P6_START['status'] = 'completed' if p6_exit == 0 else 'failed'
(P6_BACKUP / 'start.json').write_text(json.dumps(P6_START, indent=2, sort_keys=True))
if p6_exit:
    print((P6_BACKUP / 'training.log').read_text()[-12000:])
    raise RuntimeError('R2 Phase 6 failed; preserve log and inspect before rerunning')
p6_result = json.loads((ARTIFACTS_ROOT / 'training_results.json').read_text())
assert len(p6_result['models']) == 6 and p6_result['test_features_read'] is False
for source_root, target_folder in ((ARTIFACTS_ROOT, 'artifacts'), (REPORTS_ROOT, 'reports')):
    for source in source_root.rglob('*'):
        if source.is_file():
            target = P6_BACKUP / target_folder / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha256_of_file(source) == sha256_of_file(target)
p6_hashes = {str(p.relative_to(P6_BACKUP)): sha256_of_file(p) for p in P6_BACKUP.rglob('*') if p.is_file()}
(P6_BACKUP / 'sha256.json').write_text(json.dumps(p6_hashes, indent=2, sort_keys=True))
print('R2 PHASE 6 COMPLETED. Review metrics and models before Phase 7.', flush=True)
print((REPORTS_ROOT / 'phase_6_results.md').read_text(), flush=True)
print('Backup:', P6_BACKUP, flush=True)
print('STOP: no freeze or test evaluation authorized.', flush=True)
