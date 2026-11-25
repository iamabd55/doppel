#!/usr/bin/env bash
# Build data/doppel_code.zip: a git archive of code needed on Colab.
# Excludes data, results, web demo, and dev-only scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="data/doppel_code.zip"
mkdir -p data

# Save current git commit hash to COMMIT file
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || cat COMMIT 2>/dev/null || echo "unknown")
echo "$GIT_COMMIT" > COMMIT

# Base targets in git
python3 -c "
import zipfile
from pathlib import Path

out_path = Path('$OUT')
with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for base in ['detector', 'configs', 'tests']:
        for p in sorted(Path(base).rglob('*')):
            if p.is_file() and not any(part.startswith('.') or part == '__pycache__' for part in p.parts):
                zf.write(p, f'doppel/{p}')
    for extra in ['pyproject.toml', 'COMMIT', 'docs/phase4_targets.md', 'docs/phase5_targets.md', 'docs/phase5_colab_guide.md', 'scripts/train_cnn.py', 'scripts/eval_cnn.py', 'scripts/eval_phase5.py', 'scripts/select_threshold.py', 'scripts/diagnose_fa.py', 'scripts/rescore_checkpoints.py', 'scripts/seam_test_a3.py', 'scripts/verify_real_dev_b4.py', 'notebooks/phase5_train.ipynb']:
        p = Path(extra)
        if p.is_file():
            zf.write(p, f'doppel/{p}')
"

SIZE=$(du -h "$OUT" | cut -f1)
echo "Created $OUT ($SIZE)"
