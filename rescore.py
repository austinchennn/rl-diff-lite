"""Re-score saved generations with the current extract() (same rule for both models)."""
import json
import sys

from common import RESULTS
from gen_eval import extract, same

size = sys.argv[1] if len(sys.argv) > 1 else "1.5b"
p = RESULTS / f"gen_eval_{size}.json"
d = json.load(open(p))
for tag, v in d.items():
    for r in v["records"]:
        r["pred"], r["text"] = extract(r["text"])
        r["ok"] = same(r["pred"], r["gold"])
        r["n_tokens_kept"] = len(r["text"].split())
    v["acc"] = sum(r["ok"] for r in v["records"]) / len(v["records"])
    v["mean_words_kept"] = sum(r["n_tokens_kept"] for r in v["records"]) / len(v["records"])
json.dump(d, open(p, "w"), indent=1)
print({k: {"acc": v["acc"], "mean_words_kept": round(v["mean_words_kept"], 1)} for k, v in d.items()})
