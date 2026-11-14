"""Where in the weights did RL write? Compare a base model with its RL-from-base checkpoint.

Per tensor: relative Frobenius change ||dW||/||W||, the fraction of entries left bit-identical
in bf16 (the base checkpoint's storage precision), and, for 2-D matrices, the stable rank of dW.
"""
import argparse
import glob
import json
import re
from collections import defaultdict

import torch
from safetensors import safe_open

from common import PAIRS, RESULTS, local_path


def load(path):
    out = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                out[k] = fh.get_tensor(k)
    return out


def module_type(name):
    for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
              "input_layernorm", "post_attention_layernorm"]:
        if t in name:
            return t
    return name.split(".")[-2] if "." in name else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.5b", choices=PAIRS)
    ap.add_argument("--no-rank", action="store_true")
    a = ap.parse_args()
    base_id, rl_id = PAIRS[a.size]
    base, rl = load(local_path(base_id)), load(local_path(rl_id))
    print("base dtype", next(iter(base.values())).dtype, "| rl dtype", next(iter(rl.values())).dtype)

    rows = []
    for name, wb in base.items():
        rname = name
        if rname not in rl:
            print("missing in RL:", name)
            continue
        wr = rl[rname]
        b32, r32 = wb.float(), wr.float()
        d = r32 - b32
        row = {
            "name": name,
            "layer": int(m.group(1)) if (m := re.search(r"layers\.(\d+)\.", name)) else -1,
            "type": module_type(name),
            "numel": wb.numel(),
            "rel_change": (d.norm() / b32.norm()).item(),
            "frac_identical_bf16": (wr.to(torch.bfloat16) == wb.to(torch.bfloat16)).float().mean().item(),
        }
        if d.ndim == 2 and not a.no_rank and "embed" not in name:
            s = torch.linalg.svdvals(d)
            row["stable_rank"] = (s.pow(2).sum() / s[0].pow(2)).item()
            row["rank_99"] = int((s.pow(2).cumsum(0) / s.pow(2).sum() < 0.99).sum().item()) + 1
            row["full_dim"] = min(d.shape)
        rows.append(row)
    if "lm_head.weight" in rl and "lm_head.weight" not in base:
        wr, wb = rl["lm_head.weight"], base["model.embed_tokens.weight"]
        rows.append({"name": "lm_head.weight (vs tied embed)", "layer": -1, "type": "lm_head", "numel": wb.numel(),
                     "rel_change": ((wr.float() - wb.float()).norm() / wb.float().norm()).item(),
                     "frac_identical_bf16": (wr.to(torch.bfloat16) == wb.to(torch.bfloat16)).float().mean().item()})

    tot = sum(r["numel"] for r in rows)
    ident = sum(r["numel"] * r["frac_identical_bf16"] for r in rows)
    by_type = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        t = by_type[r["type"]]
        t[0] += r["numel"]; t[1] += r["numel"] * r["frac_identical_bf16"]; t[2] += r["rel_change"] * r["numel"]
    summary = {
        "pair": [base_id, rl_id],
        "total_params_compared": tot,
        "overall_frac_identical_bf16": ident / tot,
        "by_type": {k: {"frac_identical_bf16": v[1] / v[0], "mean_rel_change": v[2] / v[0]} for k, v in by_type.items()},
    }
    json.dump({"summary": summary, "rows": rows}, open(RESULTS / f"weight_diff_{a.size}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
