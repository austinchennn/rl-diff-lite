"""Shared helpers: model ids, download, and GSM8K prompts."""
import json
from pathlib import Path

PAIRS = {
    "1.5b": ("Qwen/Qwen2.5-1.5B", "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo"),
    "0.5b": ("Qwen/Qwen2.5-0.5B", "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo"),
}
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def local_path(repo_id: str) -> str:
    """Checkpoints are fetched into ./models by download.py."""
    p = ROOT / "models" / repo_id
    assert any(p.glob("*.safetensors")), f"run download.py first: {p}"
    return str(p)


def gsm8k(n: int):
    rows = [json.loads(l) for l in open(RESULTS / "gsm8k_test.jsonl")]
    return rows[:n]


def prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: Let's think step by step."


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")
