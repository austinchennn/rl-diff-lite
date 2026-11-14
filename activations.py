"""Where in the forward pass do the base and RL models diverge, on identical inputs?

Both models read the same GSM8K prompts (teacher-forced, no generation). Per layer we compare
the residual-stream hidden states: cosine similarity, relative distance, and linear CKA across
prompts. We also compare next-token distributions over all prompt positions.
"""
import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import PAIRS, RESULTS, gsm8k, local_path, prompt


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    return (hsic / ((x.T @ x).norm() * (y.T @ y).norm())).item()


@torch.no_grad()
def run(model, tok, texts, dev):
    last, mean, logps = [], [], []
    for t in texts:
        ids = tok(t, return_tensors="pt").to(dev)
        out = model(**ids, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, 0].float().cpu()  # (L+1, T, d)
        last.append(hs[:, -1])
        mean.append(hs.mean(1))
        logps.append(F.log_softmax(out.logits[0].float(), -1).cpu())
    return torch.stack(last, 1), torch.stack(mean, 1), logps  # (L+1, N, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.5b", choices=PAIRS)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    base_id, rl_id = PAIRS[a.size]
    texts = [prompt(r["question"]) for r in gsm8k(a.n)]
    tok = AutoTokenizer.from_pretrained(local_path(base_id))

    res = {}
    for tag, rid in [("base", base_id), ("rl", rl_id)]:
        m = AutoModelForCausalLM.from_pretrained(local_path(rid), dtype=torch.float32).to(dev).eval()
        res[tag] = run(m, tok, texts, dev)
        del m
        if dev == "mps":
            torch.mps.empty_cache()

    (bl, bm, blp), (rl, rm, rlp) = res["base"], res["rl"]
    layers = []
    for l in range(bl.shape[0]):
        layers.append({
            "layer": l,
            "cos_last": F.cosine_similarity(bl[l], rl[l], dim=-1).mean().item(),
            "reldist_last": ((rl[l] - bl[l]).norm(dim=-1) / bl[l].norm(dim=-1)).mean().item(),
            "cka_last": linear_cka(bl[l], rl[l]),
            "cka_mean": linear_cka(bm[l], rm[l]),
        })
    kl = torch.cat([(r.exp() * (r - b)).sum(-1) for b, r in zip(blp, rlp)])
    top1 = torch.cat([(b.argmax(-1) == r.argmax(-1)).float() for b, r in zip(blp, rlp)])
    summary = {"pair": [base_id, rl_id], "n_prompts": a.n, "n_positions": int(kl.numel()),
               "mean_kl_rl_vs_base": kl.mean().item(), "median_kl": kl.median().item(),
               "top1_agreement": top1.mean().item()}
    json.dump({"summary": summary, "layers": layers}, open(RESULTS / f"activations_{a.size}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in layers:
        print(f"L{r['layer']:2d} cos={r['cos_last']:.4f} reldist={r['reldist_last']:.4f} cka_last={r['cka_last']:.4f} cka_mean={r['cka_mean']:.4f}")


if __name__ == "__main__":
    main()
