#!/usr/bin/env bash
# Build data/doppel_code.zip: a git archive of code needed on Colab.
# Excludes data, results, web demo, and dev-only scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="data/doppel_code.zip"
mkdir -p data

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
    for extra in ['pyproject.toml', 'docs/phase4_targets.md', 'scripts/train_cnn.py', 'scripts/eval_cnn.py', 'scripts/select_threshold.py', 'scripts/diagnose_fa.py', 'scripts/rescore_checkpoints.py']:
        p = Path(extra)
        if p.is_file():
            zf.write(p, f'doppel/{p}')
"

SIZE=$(du -h "$OUT" | cut -f1)
echo "Created $OUT ($SIZE)"
