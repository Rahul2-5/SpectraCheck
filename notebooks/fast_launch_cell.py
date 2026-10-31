# Launches Phase 6 only. The runner independently backs up all six completed models/results.
assert not ('fast_process' in globals() and fast_process.poll() is None), 'Fast training is already running'
assert not any('scripts/train_models.py' in line or 'scripts/run_fast_training.py' in line for line in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()), 'Another training process is active'
FAST_LAUNCH_LOG = Path('/content/drive/MyDrive/aidetect/r2') / ('fast_launch_' + __import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.log')
with FAST_LAUNCH_LOG.open('x') as stream:
    fast_process = subprocess.Popen([sys.executable, '-u', 'scripts/run_fast_training.py', '--config', str(FAST_CONFIG)], cwd=PROJECT_DIR, env=fast_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
print('Fast Phase 6 runner PID:', fast_process.pid)
print('Launch log:', FAST_LAUNCH_LOG)
print('STOP: no freeze/test evaluation queued. Use the status cell; do not rerun launch.')
