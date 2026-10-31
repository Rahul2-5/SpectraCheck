# Read-only status; safe while the training subprocess is running.
from pathlib import Path
import json
root = Path('/content/drive/MyDrive/aidetect/r2')
for folder in sorted((root / 'review_backup').glob('r2_fast*_training_*'))[-2:]:
    print(folder)
    for name in ('start.json', 'training.log'):
        path = folder / name
        if path.is_file(): print(name, path.read_text()[-3000:])
for folder in sorted(root.glob('r2_fast*_phase6_checkpoints')):
    path = folder / 'status.json'
    if path.is_file(): print(folder, path.read_text())
print('STOP before Phase 7. Review actual completed outputs and practical target first.')
