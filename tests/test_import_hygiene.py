"""Import hygiene: the numpy-facing modules must not pull in qiskit.

Each module is imported in a fresh subprocess and 'qiskit' must be absent from
sys.modules afterwards. Only ``circuit`` and ``qnn`` (loaded lazily by the
quantum package) may import qiskit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

QISKIT_FREE_MODULES = [
    "traqmania.config",
    "traqmania.agents.base",
    "traqmania.agents.classical.mlp",
    "traqmania.agents.training.dqn",
    "traqmania.agents.quantum",  # package only; circuit/qnn are lazy submodules
    "traqmania.agents.quantum.fastsim",
    "traqmania.agents.quantum.adjoint",
    "traqmania.agents.quantum.qdqn",
    "traqmania.agents.training.spsa",
    "traqmania.hardware",  # qiskit/qiskit-ibm-runtime imports live inside functions
]

# Environment modules may not exist yet in every worktree; skip them if absent.
ENV_MODULES = [
    "traqmania.env",
    "traqmania.env.racing_env",
]

_CHECK = (
    "import {module}, sys; "
    "offenders = sorted(m for m in sys.modules if m == 'qiskit' or m.startswith('qiskit.')); "
    "assert 'qiskit' not in sys.modules, "
    "'importing {module} pulled in qiskit: %s' % offenders"
)


def _assert_import_is_qiskit_free(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHECK.format(module=module)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing {module} in a fresh interpreter failed or imported qiskit:\n"
        f"{result.stderr}"
    )


@pytest.mark.parametrize("module", QISKIT_FREE_MODULES)
def test_module_import_is_qiskit_free(module: str):
    _assert_import_is_qiskit_free(module)


@pytest.mark.parametrize("module", ENV_MODULES)
def test_env_import_is_qiskit_free(module: str):
    pytest.importorskip(module)
    _assert_import_is_qiskit_free(module)
