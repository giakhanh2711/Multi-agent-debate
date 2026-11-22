import re
from pathlib import Path
import json


def normalize_cf_cookie(raw):
    raw = raw.strip().strip('"').strip("'")
    match = re.search(r"CF_Authorization=([^;\n]+)", raw)
    return f"CF_Authorization={match.group(1)}" if match else (
        f"CF_Authorization={raw}" if raw.startswith("eyJ") else raw
    )


def reformat_string(answer: str):
    return re.sub(r"\s+", " ", answer.casefold()).strip().rstrip(" .")


def append_to_json(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_correct(prediction: str, aliases) -> bool:
    pred = reformat_string(prediction)
    if not pred:
        return False
    return any(pred == reformat_string(alias) for alias in aliases if alias)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
