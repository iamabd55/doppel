#!/usr/bin/env bash
# Build data/doppel_code.zip: a git archive of code needed on Colab.
# Excludes data, results, web demo, and dev-only scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="data/doppel_code.zip"
mkdir -p data

# Base targets in git
PATHS=(detector configs tests pyproject.toml docs/phase4_targets.md)

# Include training-related scripts if they exist in HEAD
for script in scripts/train_cnn.py scripts/select_threshold.py; do
    if git ls-tree -r --name-only HEAD | grep -qx "$script"; then
        PATHS+=("$script")
    fi
done

git archive HEAD --prefix=doppel/ -o "$OUT" "${PATHS[@]}"

# Ensure working directory training scripts or cnn files are bundled if present locally
for extra in scripts/train_cnn.py scripts/select_threshold.py detector/cnn/unet.py detector/cnn/corr_net.py detector/cnn/infer.py detector/cnn/__init__.py; do
    if [ -f "$extra" ]; then
        python3 -c "
import zipfile
with zipfile.ZipFile('$OUT', 'a') as zf:
    names = zf.namelist()
    target = 'doppel/$extra'
    if target not in names:
        zf.write('$extra', target)
"
    fi
done

SIZE=$(du -h "$OUT" | cut -f1)
echo "Created $OUT ($SIZE)"
