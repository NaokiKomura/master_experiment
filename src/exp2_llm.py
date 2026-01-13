from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .exp2_prompts import PromptPack


@dataclass(frozen=True)
class LLMResult:
    label: int
    confidence: float
    reason: str
    raw_text: str


def _safe_json_loads(s: str) -> dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 先頭〜末尾のJSONブロックを救出（余計な前後文が出た場合）
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


def _coerce_label(x: Any) -> int:
    if isinstance(x, bool):
        raise ValueError("label cannot be bool")
    if isinstance(x, (int, float)) and int(x) == x:
        v = int(x)
    elif isinstance(x, str):
        v = int(x.strip())
    else:
        raise ValueError(f"Invalid label type: {type(x)}")
    if v < 0 or v > 4:
        raise ValueError(f"Label out of range: {v}")
    return v


def create_client(api_key: str, api_base: Optional[str] = None) -> OpenAI:
    if api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    return OpenAI(api_key=api_key)


def classify(
    client: OpenAI,
    model: str,
    prompt: PromptPack,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 3,
    retry_sleep_sec: float = 2.0,
) -> LLMResult:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
            )
            text = resp.choices[0].message.content or ""
            obj = _safe_json_loads(text)

            label = _coerce_label(obj.get("label"))
            confidence = float(obj.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(obj.get("reason", "")).strip()

            return LLMResult(label=label, confidence=confidence, reason=reason, raw_text=text)

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep_sec * attempt)
            else:
                break

    raise RuntimeError(f"LLM call failed after {max_retries} retries. last_err={last_err}")