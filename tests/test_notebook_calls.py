"""Unit tests validating all script calls, CLI arguments, and dataset kwargs in the notebook."""

import ast
import inspect
import shlex
from pathlib import Path
import nbformat
import pytest

from detector.colab_dataset import ColabDoppelDataset
from scripts import diagnose_fa, eval_cnn, select_threshold, train_cnn

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "phase4_train.ipynb"


def _clean_python_code(source: str) -> str:
    """Remove IPython shell lines and multi-line continuations for AST parsing."""
    cleaned = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%"):
            while lines[i].strip().endswith("\\") and i + 1 < len(lines):
                i += 1
            i += 1
            continue
        cleaned.append(line)
        i += 1
    return "\n".join(cleaned)


def _load_notebook():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_colab_doppel_dataset_kwargs():
    """Verify all ColabDoppelDataset calls in the notebook match the real __init__ signature."""
    nb = _load_notebook()
    sig = inspect.signature(ColabDoppelDataset.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    call_count = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code = _clean_python_code(cell.source)
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, "id", getattr(node.func, "attr", None))
                if func_name == "ColabDoppelDataset":
                    call_count += 1
                    for kw in node.keywords:
                        assert kw.arg in valid_params, (
                            f"Cell {idx}: unexpected keyword argument '{kw.arg}' "
                            f"in ColabDoppelDataset call. Valid: {sorted(valid_params)}"
                        )

    assert call_count >= 1, "Expected at least one ColabDoppelDataset call in notebook."


def test_train_cnn_command_args():
    """Verify all train_cnn.py invocations in notebook parse successfully without unknown flags."""
    nb = _load_notebook()
    train_cmds = []

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code = _clean_python_code(cell.source)
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                elts = node.value.elts
                if len(elts) >= 2 and isinstance(elts[0], ast.Constant) and elts[0].value == "python3":
                    second_elt = elts[1]
                    # Check if second element references train_cnn.py
                    is_train = False
                    if isinstance(second_elt, ast.Constant) and "train_cnn.py" in str(second_elt.value):
                        is_train = True
                    elif isinstance(second_elt, ast.Call) and "train_cnn.py" in ast.unparse(second_elt):
                        is_train = True

                    if is_train:
                        args = []
                        for elt in elts[2:]:
                            if isinstance(elt, ast.Constant):
                                args.append(str(elt.value))
                            else:
                                if args and args[-1] == "--model":
                                    args.append("corr")
                                else:
                                    args.append("dummy_val")
                        train_cmds.append((idx, args))

    assert len(train_cmds) >= 4, f"Expected at least 4 train_cnn calls, found {len(train_cmds)}"

    for cell_idx, args in train_cmds:
        try:
            parsed = train_cnn.parse_args(args)
            assert parsed.model in ["unet", "corr"]
        except SystemExit as e:
            pytest.fail(f"Cell {cell_idx} train_cnn command failed parse_args with error: {e}")


def test_eval_cnn_command_args():
    """Verify all eval_cnn.py shell commands in notebook parse successfully without unknown flags."""
    nb = _load_notebook()
    eval_cmds = []

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("!python3"):
                cmd_str = line[len("!python3"):].strip()
                while cmd_str.endswith("\\") and i + 1 < len(lines):
                    i += 1
                    cmd_str = cmd_str[:-1].strip() + " " + lines[i].strip()
                tokens = shlex.split(cmd_str)
                if tokens and "eval_script" in tokens[0]:
                    args = []
                    for t in tokens[1:]:
                        if t.startswith("$") or t.startswith("/content"):
                            if args and args[-1] == "--model":
                                args.append("corr")
                            else:
                                args.append("dummy_val")
                        else:
                            args.append(t)
                    eval_cmds.append((idx, args))
            i += 1

    assert len(eval_cmds) >= 3, f"Expected at least 3 eval_cnn calls in Cell 11, found {len(eval_cmds)}"

    for cell_idx, args in eval_cmds:
        try:
            parsed = eval_cnn.parse_args(args)
            assert parsed.checkpoint == "dummy_val"
        except SystemExit as e:
            pytest.fail(f"Cell {cell_idx} eval_cnn command failed parse_args with error: {e}")


def test_select_threshold_command_args():
    """Verify all select_threshold.py shell commands in notebook parse successfully."""
    nb = _load_notebook()
    select_cmds = []

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("!python3"):
                cmd_str = line[len("!python3"):].strip()
                while cmd_str.endswith("\\") and i + 1 < len(lines):
                    i += 1
                    cmd_str = cmd_str[:-1].strip() + " " + lines[i].strip()
                tokens = shlex.split(cmd_str)
                if tokens and "select_script" in tokens[0]:
                    args = []
                    for t in tokens[1:]:
                        if t.startswith("$") or t.startswith("/content"):
                            if args and args[-1] == "--model":
                                args.append("corr")
                            else:
                                args.append("dummy_val")
                        else:
                            args.append(t)
                    select_cmds.append((idx, args))
            i += 1

    assert len(select_cmds) >= 1, "Expected select_threshold call in notebook Cell 12."

    for cell_idx, args in select_cmds:
        try:
            parsed = select_threshold.parse_args(args)
            assert parsed.checkpoint == "dummy_val"
        except SystemExit as e:
            pytest.fail(f"Cell {cell_idx} select_threshold command failed parse_args with error: {e}")


def test_diagnose_fa_command_args():
    """Verify all diagnose_fa.py shell commands in notebook parse successfully."""
    nb = _load_notebook()
    diag_cmds = []

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("!python3"):
                cmd_str = line[len("!python3"):].strip()
                while cmd_str.endswith("\\") and i + 1 < len(lines):
                    i += 1
                    cmd_str = cmd_str[:-1].strip() + " " + lines[i].strip()
                tokens = shlex.split(cmd_str)
                if tokens and "diag_script" in tokens[0]:
                    args = []
                    for t in tokens[1:]:
                        if t.startswith("$") or t.startswith("/content"):
                            if args and args[-1] == "--model":
                                args.append("corr")
                            else:
                                args.append("dummy_val")
                        else:
                            args.append(t)
                    diag_cmds.append((idx, args))
            i += 1

    assert len(diag_cmds) >= 1, "Expected diagnose_fa call in notebook."

    for cell_idx, args in diag_cmds:
        try:
            parsed = diagnose_fa.parse_args(args)
            assert parsed.checkpoint == "dummy_val"
        except SystemExit as e:
            pytest.fail(f"Cell {cell_idx} diagnose_fa command failed parse_args with error: {e}")

