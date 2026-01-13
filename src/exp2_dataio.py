from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .exp2_prompts import FewshotExample


def read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def read_research_data_csv(path: str | Path) -> pd.DataFrame:
    """
    research_data.csv 想定:
      - copy_candidates: 入力広告コピー
      - Label: 正解ラベル(0-4)
      - id: 任意（なければ付与）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    df = pd.read_csv(p)

    required = {"copy_candidates", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. required={required}")

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})

    df["id"] = df["id"].astype(str)
    df["copy_candidates"] = df["copy_candidates"].astype(str)
    df["Label"] = df["Label"].astype(int)

    # バリデーション
    if not df["Label"].between(0, 4).all():
        bad = df.loc[~df["Label"].between(0, 4), ["id", "Label"]].head(10)
        raise ValueError(f"Label must be in [0,4]. Bad rows (head):\n{bad}")

    return df


def read_fewshot_examples_jsonl(path: str | Path) -> list[FewshotExample]:
    """
    data/exp1_representatives.jsonl 想定（実験1で確定した代表例）:
      {"label":1,"text":"...","rationale":"..."}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Few-shot examples not found: {p}")

    examples: list[FewshotExample] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for k in ("label", "text", "rationale"):
            if k not in obj:
                raise ValueError(f"Missing key '{k}' at line {i} in {p}")

        label = int(obj["label"])
        if label < 0 or label > 4:
            raise ValueError(f"label must be 0-4 at line {i}, got {label}")

        examples.append(
            FewshotExample(
                label=label,
                text=str(obj["text"]),
                rationale=str(obj["rationale"]),
            )
        )

    if not examples:
        raise ValueError(f"No examples loaded from {p}")
    return examples


def auto_sample_examples_from_dataset(
    df: pd.DataFrame,
    *,
    per_label: int = 2,
    text_col: str = "copy_candidates",
    label_col: str = "Label",
) -> list[FewshotExample]:
    """
    代表例ファイルがない場合のフォールバック。
    rationale は空にし、実験1由来の「解説」相当は含まれません（論文要件的には非推奨）。
    """
    out: list[FewshotExample] = []
    for lab in [0, 1, 2, 3, 4]:
        sub = df[df[label_col] == lab].head(per_label)
        for _, r in sub.iterrows():
            out.append(FewshotExample(label=int(lab), text=str(r[text_col]), rationale=""))
    return out