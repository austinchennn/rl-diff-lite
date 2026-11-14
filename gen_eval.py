"""Sanity check that RL changed behaviour: greedy GSM8K accuracy and response length, same prompt for both."""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import PAIRS, RESULTS, gold, gsm8k, local_path, prompt


STOP = re.compile(r"\n\s*\[?(Question|Q)\]?\s*[:\]]|\n\s*\[Answer\]|\n\s*(Human|User)\s*:|\n{3,}")


def extract(text):
    """Cut where the model starts inventing a new question, then take the stated answer."""
    text = STOP.split(text)[0]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    stated = re.findall(r"answer is[:\s$]*(-?[\d,]*\.?\d+)", text)
    cand = boxed[-1] if boxed else (stated[-1] if stated else text)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", cand)
    return nums[-1].replace(",", "").rstrip(".") if nums else None, text


def same(p, g):
    try:
        return p is not None and abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.5b", choices=PAIRS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = gsm8k(a.n)
    out = {}
    for tag, rid in zip(["base", "rl"], PAIRS[a.size]):
        tok = AutoTokenizer.from_pretrained(local_path(rid), padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(local_path(rid), dtype=torch.bfloat16).to(dev).eval()
        recs = []
        for i in range(0, len(rows), a.bs):
            batch = rows[i:i + a.bs]
            enc = tok([prompt(r["question"]) for r in batch], return_tensors="pt", padding=True).to(dev)
            gen = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
                txt = tok.decode(g, skip_special_tokens=True)
                pred, cut = extract(txt)
                recs.append({"gold": gold(r["answer"]), "pred": pred, "ok": same(pred, gold(r["answer"])),
                             "n_tokens": len(tok(cut)["input_ids"]), "text": cut})
            print(tag, i + len(batch), "acc so far", sum(x["ok"] for x in recs) / len(recs), flush=True)
        out[tag] = {"acc": sum(x["ok"] for x in recs) / len(recs),
                    "mean_tokens": sum(x["n_tokens"] for x in recs) / len(recs), "records": recs}
        del m
        if dev == "mps":
            torch.mps.empty_cache()
    json.dump(out, open(RESULTS / f"gen_eval_{a.size}.json", "w"), indent=1)
    print({k: {"acc": v["acc"], "mean_tokens": v["mean_tokens"]} for k, v in out.items()})


if __name__ == "__main__":
    main()
