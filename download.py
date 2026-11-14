"""Download the four checkpoints into ./models (curl with resume and retries; no hub client needed)."""
import json
import subprocess
import sys
import urllib.request

from common import PAIRS, ROOT

sizes = sys.argv[1:] or list(PAIRS)
for size in sizes:
    for repo in PAIRS[size]:
        d = ROOT / "models" / repo
        d.mkdir(parents=True, exist_ok=True)
        meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}"))
        for s in meta["siblings"]:
            f = s["rfilename"]
            if not f.endswith((".json", ".safetensors", ".txt")) or (d / f).exists():
                continue
            part = d / (f + ".part")
            for attempt in range(20):
                r = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--retry", "5", "--retry-all-errors",
                                    "-o", str(part), f"https://huggingface.co/{repo}/resolve/main/{f}"])
                if r.returncode == 0:
                    break
            else:
                sys.exit(f"failed: {repo}/{f}")
            part.rename(d / f)
        print(repo, round(sum(p.stat().st_size for p in d.iterdir()) / 1e9, 2), "GB", flush=True)
