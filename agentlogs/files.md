# File Inventory

Summary of files in the Doppel repository:

## Root Configuration and Documentation
- `DOPPEL_AGENT.md`: Primary briefing and rulebook for autonomous and assisted development on Doppel. Contains rules, architecture, frozen baseline numbers, and phase roadmaps.
- `README.md`: Public project documentation, including problem statement, methodology comparison, status table, and baseline benchmarks.
- `pyproject.toml`: Python project configuration specifying dependencies (Flask, NumPy, OpenCV, scikit-image, scikit-learn, SciPy) and Python version requirements.
- `uv.lock`: Dependency lockfile managed by `uv`.
- `.gitignore`: Specifies ignored directories including `.venv`, model weights, and local data.

## Core Detection Module (`detector/`)
- `detector/__init__.py`: Package initialization.
- `detector/detect.py`: Top-level detection pipeline. Orchestrates feature extraction, lexicographic matching, shift vector voting, block painting, morphological closing, and small component removal.
- `detector/features.py`: Feature extraction using sliding-window 8x8 block decomposition, standard deviation texture filtering, 2D orthonormal DCT, zigzag coefficient ordering, and quantization.
- `detector/matching.py`: Candidate matching via lexicographic sorting (`np.lexsort`), neighbor window search, Euclidean distance thresholding, and canonical shift vector orientation.
- `detector/evaluate.py`: Evaluation utilities computing pixel-level true positives, false positives, false negatives, precision, recall, and F1 score against ground truth masks.

## Evaluation and Utility Scripts (`scripts/`)
- `scripts/batch_test.py`: Benchmark evaluation script running the DCT detector on CoMoFoD images 001-040 across categories, evaluating against full mask M and paste-only mask B.
- `scripts/false_alarm_test.py`: False alarm evaluation testing untouched original images (001_O through 040_O) to verify zero or low false positive rates.
- `scripts/experiments.py`: Robustness experimentation script measuring F1 resilience against Gaussian noise, Gaussian blur, and JPEG compression degradation.
- `scripts/inspect_comofod.py`: Diagnostic utility comparing pixel differences between forged (F), original (O), and ground truth masks (B, M) for CoMoFoD set 001.
- `scripts/make_test.py`: Simple synthetic generation script producing a translated copy-paste test image and mask.
- `scripts/run.py`: Quick runner script evaluating the detector on a synthetic sample image.
- `scripts/try_001.py`: Parameter exploration script testing the effect of different `min_votes` thresholds on CoMoFoD set 001.

## Interactive Web Baseline Demo (`web/`)
- `web/app.py`: Flask application providing an interactive web interface and REST API endpoints (`/api/detect`, `/api/comofod/list`, `/api/comofod/<num>`).
- `web/templates/index.html`: Web dashboard HTML template.
- `web/static/app.js`: Client-side interaction logic for image upload, parameter control, and result visualization.
- `web/static/style.css`: Dashboard styling.
- `web/static/lenis.min.js`: Smooth scrolling library.
- `web/static/lenis.css`: Styles for smooth scrolling.

## Learning and Visuals
- `learning/dct_demo.py`: Educational script illustrating 1D and 2D DCT fundamentals, energy compaction, and inverse reconstruction.
- `figures/robustness.png`: Performance chart showing F1 score under noise, blur, and JPEG compression.

## Agent Documentation (`agentlogs/`)
- `agentlogs/architecture.md`: Architectural layout, data flow, and pipeline progression across phases.
- `agentlogs/files.md`: Inventory and responsibility summary of every file in the repository.
- `agentlogs/changelog.md`: Chronological log of changes, phase updates, and task milestones.

## Data and Results Directories (Untracked / Outputs)
- `data/`: Contains CoMoFoD benchmark images and local synthetic test sets.
- `results/`: Stores experiment CSV tables, diagnostic plots, and markdown summaries per phase.
