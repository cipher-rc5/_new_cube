"""Tests for the ``--backend`` routing in cube3d/generate.py.

generate.py builds its argparse parser inline under ``if __name__ ==
"__main__":`` rather than in a reusable function, so we cannot import and call a
parser factory. Running the module would also require torch and model weights.

To validate the documented invariants without executing generation, we parse
generate.py's source with the ``ast`` module and assert on the literal argparse
configuration and the backend-routing branches. This needs no third-party
dependencies (not even torch), so it runs everywhere in CI.

Invariants asserted:
* a ``--backend`` argument exists with choices ["torch", "mlx"] and default
  "torch";
* the mlx branch imports MlxEngine and guards on MLX_AVAILABLE + mps device;
* fast-inference is CUDA-only and falls back to the standard Engine otherwise.
"""

import ast
import os

from tests.conftest import REPO_ROOT

GENERATE_PY = os.path.join(REPO_ROOT, "cube3d", "generate.py")


def _read_source() -> str:
    with open(GENERATE_PY, "r", encoding="utf-8") as f:
        return f.read()


def _find_backend_add_argument(tree: ast.AST):
    """Return the ast.Call node for parser.add_argument("--backend", ...)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            if node.args[0].value == "--backend":
                return node
    return None


def test_generate_source_parses():
    """generate.py must be syntactically valid Python."""
    ast.parse(_read_source())


def test_backend_argument_choices_and_default():
    """--backend exists with choices ["torch", "mlx"] and default "torch"."""
    tree = ast.parse(_read_source())
    call = _find_backend_add_argument(tree)
    assert call is not None, "No parser.add_argument('--backend', ...) found"

    kwargs = {kw.arg: kw.value for kw in call.keywords}

    assert "choices" in kwargs, "--backend must declare choices"
    choices = ast.literal_eval(kwargs["choices"])
    assert list(choices) == ["torch", "mlx"], choices

    assert "default" in kwargs, "--backend must declare a default"
    default = ast.literal_eval(kwargs["default"])
    assert default == "torch", default


def test_default_backend_is_torch_in_source():
    """The literal default backend string is torch (belt-and-suspenders)."""
    src = _read_source()
    assert 'default="torch"' in src or "default='torch'" in src


def test_mlx_branch_guards_and_imports():
    """The mlx branch imports MlxEngine/MLX_AVAILABLE and guards on mps."""
    src = _read_source()
    assert 'args.backend == "mlx"' in src
    assert "from cube3d.inference.mlx_engine import" in src
    assert "MLX_AVAILABLE" in src
    # Must require an Apple GPU (mps) device for the mlx backend.
    assert 'device.type != "mps"' in src


def test_fast_inference_is_cuda_only_with_fallback():
    """fast-inference routes to EngineFast only on cuda, else falls back."""
    src = _read_source()
    # EngineFast is selected only when fast_inference AND device is cuda.
    assert 'args.fast_inference and device.type == "cuda"' in src
    assert "EngineFast(" in src
    # Standard Engine is the fallback path.
    assert "fast-inference is CUDA-only" in src
    assert "engine = Engine(" in src
