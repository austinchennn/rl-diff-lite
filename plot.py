"""Figures from results/*.json."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS

size = sys.argv[1] if len(sys.argv) > 1 else "1.5b"
w = json.load(open(RESULTS / f"weight_diff_{size}.json"))
act_path = RESULTS / f"activations_{size}.json"
act = json.load(open(act_path)) if act_path.exists() else None

types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
L = 1 + max(r["layer"] for r in w["rows"])
grid = np.full((len(types), L), np.nan)
for r in w["rows"]:
    if r["type"] in types and r["name"].endswith("weight"):
        grid[types.index(r["type"]), r["layer"]] = r["rel_change"]

ncols = 2 if act else 1
fig, ax = plt.subplots(1, ncols, figsize=(6.5 * ncols, 3.8))
ax = np.atleast_1d(ax)
im = ax[0].imshow(grid * 100, aspect="auto", cmap="viridis")
ax[0].set_yticks(range(len(types)), types)
ax[0].set_xlabel("layer")
ax[0].set_title("Relative weight change ||ΔW|| / ||W|| (%)")
fig.colorbar(im, ax=ax[0])
if act:
    ls = act["layers"]
    x = [r["layer"] for r in ls]
    ax[1].plot(x, [1 - r["cka_last"] for r in ls], label="1 − CKA (last token)")
    ax[1].plot(x, [1 - r["cos_last"] for r in ls], label="1 − cosine (last token)")
    ax[1].plot(x, [r["reldist_last"] for r in ls], label="relative distance")
    ax[1].set_xlabel("layer (0 = embeddings)")
    ax[1].set_yscale("log")
    ax[1].set_title("Base vs RL hidden states on the same prompts")
    ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / f"fig_{size}.png", dpi=150)
print("saved", RESULTS / f"fig_{size}.png")
