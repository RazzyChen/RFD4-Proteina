"""Build-time smoke test: imports, TMol operators for the selected profile, Inductor."""

import os
from importlib.metadata import distributions

import atomworks  # noqa: F401
import torch
from openbabel import pybel
from tmol.score.ljlk.potentials._compiled import ljlk_pose_scores

import rfproteina  # noqa: F401

accelerator = os.environ["ACCELERATOR"]
assert torch.__version__.endswith("+" + {"cu13": "cu130", "cu12": "cu126", "cpu": "cpu"}[accelerator])
assert pybel.readstring("smi", "CCO").OBMol.NumAtoms() == 3
assert callable(ljlk_pose_scores)
names = {d.metadata["Name"].lower().replace("_", "-") for d in distributions()}
ops = {n for n in names if n.startswith("tmol-ops-")}
assert ops == (set() if accelerator == "cpu" else {"tmol-ops-" + accelerator}), ops
x = torch.ones(8)
torch.testing.assert_close(torch.compile(lambda t: t.sin() + 1, backend="inductor", fullgraph=True)(x), x.sin() + 1)
print("smoke test ok")
