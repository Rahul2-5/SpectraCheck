from pathlib import Path
import subprocess, sys, os, json
PROJECT_DIR = Path('/content/ai-image-fft-detector')
os.chdir(PROJECT_DIR)
FAST_CONFIG = PROJECT_DIR / 'config_r2_fast.yaml'  # CPU default; optional config_r2_fast_t4.yaml
fast_env = os.environ.copy()
fast_env['PYTHONPATH'] = str(PROJECT_DIR / 'src') + os.pathsep + str(PROJECT_DIR / 'scripts')
fast_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
# This verifies the reviewed Phase 5 record and frozen provenance; no extraction or test reads.
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
assert not any('scripts/train_models.py' in line or 'scripts/run_fast_training.py' in line for line in processes.splitlines()), 'A training process is already running'
subprocess.run([sys.executable, 'scripts/prepare_fast_training.py', '--config', str(FAST_CONFIG)], cwd=PROJECT_DIR, env=fast_env, check=True)
