from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


SYSTEM_MSG = (
    "You are a careful annotator for an academic experiment. "
    "Follow the label taxonomy strictly. "
    "Return only valid JSON with an integer label in {0,1,2,3,4}."
)


@dataclass(frozen=True)
class FewshotExample:
    label: int  # 0-4
    text: str
    rationale: str  # short explanation


@dataclass(frozen=True)
class PromptPack:
    system: str
    user: str


def build_taxonomy_block(taxonomy_md: str) -> str:
    return f"# 炎上要因分類（定義）\n{taxonomy_md}".strip()


def build_zero_shot_prompt(taxonomy_md: str, ad_copy: str) -> PromptPack:
    user = f"""
以下の「炎上要因分類（4分類）」定義に従い、広告コピー（入力文）を 0〜4 の単一ラベルで分類してください。

- 0: どの分類にも当てはまらない
- 1〜4: 炎上要因分類1〜4のいずれか（定義に従う）

{build_taxonomy_block(taxonomy_md)}

# 入力文（広告コピー）
{ad_copy}

# 出力形式（厳守）
必ず次のJSONのみを返してください（余計な文章は禁止）:
{{
  "label": <0から4の整数>,
  "confidence": <0から1の実数>,
  "reason": "<短い根拠（1-3文）>"
}}
""".strip()
    return PromptPack(system=SYSTEM_MSG, user=user)


def _format_examples(
    examples: Iterable[FewshotExample],
    max_examples_per_label: Optional[int] = 2,
) -> str:
    """
    代表例（実験1で確定した正解事例）を整形。
    """
    per_label_count: dict[int, int] = {}
    blocks = []
    for ex in examples:
        if max_examples_per_label is not None:
            c = per_label_count.get(ex.label, 0)
            if c >= max_examples_per_label:
                continue
            per_label_count[ex.label] = c + 1

        blocks.append(
            f"## ラベル: {ex.label}\n"
            f"- 例文: {ex.text}\n"
            f"- 解説: {ex.rationale}\n"
        )

    if not blocks:
        raise ValueError("No few-shot examples available after filtering.")
    return "\n".join(blocks).strip()


def build_few_shot_prompt(
    taxonomy_md: str,
    examples: list[FewshotExample],
    ad_copy: str,
    max_examples_per_label: int = 2,
) -> PromptPack:
    examples_block = _format_examples(examples, max_examples_per_label)

    user = f"""
以下の「炎上要因分類（4分類）」定義に従い、広告コピー（入力文）を 0〜4 の単一ラベルで分類してください。
参考として、実験1で確定した各ラベルの代表的な正解事例とその解説を提示します。提示例を踏まえて最も妥当なラベルを選んでください。

- 0: どの分類にも当てはまらない
- 1〜4: 炎上要因分類1〜4のいずれか（定義に従う）

{build_taxonomy_block(taxonomy_md)}

# 代表例（正解事例と解説）
{examples_block}

# 入力文（広告コピー）
{ad_copy}

# 出力形式（厳守）
必ず次のJSONのみを返してください（余計な文章は禁止）:
{{
  "label": <0から4の整数>,
  "confidence": <0から1の実数>,
  "reason": "<短い根拠（1-3文）>"
}}
""".strip()
    return PromptPack(system=SYSTEM_MSG, user=user)