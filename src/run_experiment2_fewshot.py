from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import src.config as config  # 既存config.pyを利用

from .exp2_dataio import (
    read_research_data_csv,
    read_text,
    read_fewshot_examples_jsonl,
    auto_sample_examples_from_dataset,
)
from .exp2_prompts import build_zero_shot_prompt, build_few_shot_prompt
from .exp2_llm import create_client, classify
from .exp2_eval import evaluate, save_eval


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser("Experiment2: Zero-shot vs Few-shot auto annotation")
    parser.add_argument("--dataset", default=str(getattr(config, "DATA_FILE", config.PROJECT_ROOT / "data/research_data.csv")))
    parser.add_argument("--taxonomy", default=str(config.PROJECT_ROOT / "data/taxonomy.md"))
    parser.add_argument("--fewshot", default=str(config.PROJECT_ROOT / "data/exp1_representatives.jsonl"))
    parser.add_argument("--outdir", default=str(config.PROJECT_ROOT / "data/outputs/exp2"))
    parser.add_argument("--max_examples_per_label", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fallback_autosample", action="store_true", help="if fewshot file missing, auto-sample from dataset (NOT recommended for thesis)")
    args = parser.parse_args()

    # OpenAI settings（config.pyに無い場合は.envから拾えるように）
    api_key = getattr(config, "OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Check .env and src/config.py")

    model = getattr(config, "OPENAI_MODEL", None) or "gpt-4.1-mini"
    temperature = float(getattr(config, "OPENAI_TEMPERATURE", 0.0))
    max_tokens = int(getattr(config, "OPENAI_MAX_TOKENS", 256))
    api_base = getattr(config, "OPENAI_API_BASE", None)  # optional

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = read_research_data_csv(args.dataset)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    taxonomy_md = read_text(args.taxonomy)

    # Few-shot examples
    try:
        fewshot_examples = read_fewshot_examples_jsonl(args.fewshot)
    except FileNotFoundError:
        if not args.fallback_autosample:
            raise
        fewshot_examples = auto_sample_examples_from_dataset(df, per_label=args.max_examples_per_label)

    client = create_client(api_key=api_key, api_base=api_base)

    pred_path = outdir / "predictions.csv"
    meta_path = outdir / "run_meta.json"
    eval_zero_path = outdir / "eval_zero.txt"
    eval_few_path = outdir / "eval_few.txt"

    if pred_path.exists() and not args.overwrite:
        pred_df = pd.read_csv(pred_path)
        print(f"[INFO] Loaded cached predictions: {pred_path}")
    else:
        rows = []
        for _, r in tqdm(df.iterrows(), total=len(df)):
            ad_copy = r["copy_candidates"]

            # B: Zero-shot
            p0 = build_zero_shot_prompt(taxonomy_md, ad_copy)
            res0 = classify(
                client,
                model=model,
                prompt=p0,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # P: Few-shot
            p1 = build_few_shot_prompt(
                taxonomy_md,
                examples=fewshot_examples,
                ad_copy=ad_copy,
                max_examples_per_label=args.max_examples_per_label,
            )
            res1 = classify(
                client,
                model=model,
                prompt=p1,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            rows.append(
                {
                    "id": r["id"],
                    "copy_candidates": ad_copy,
                    "gold": int(r["Label"]),
                    "pred_zero": int(res0.label),
                    "conf_zero": float(res0.confidence),
                    "reason_zero": res0.reason,
                    "pred_few": int(res1.label),
                    "conf_few": float(res1.confidence),
                    "reason_few": res1.reason,
                }
            )

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")
        print(f"[INFO] Saved predictions: {pred_path}")

    # Evaluation
    zero_summary, zero_details = evaluate(pred_df, gold_col="gold", pred_col="pred_zero")
    few_summary, few_details = evaluate(pred_df, gold_col="gold", pred_col="pred_few")

    save_eval(eval_zero_path, zero_summary, zero_details)
    save_eval(eval_few_path, few_summary, few_details)

    # Run metadata
    delta = {
        "n": int(zero_summary.n),
        "zero": zero_summary.__dict__,
        "few": few_summary.__dict__,
        "diff_few_minus_zero": {
            "accuracy": few_summary.accuracy - zero_summary.accuracy,
            "macro_f1": few_summary.macro_f1 - zero_summary.macro_f1,
            "weighted_f1": few_summary.weighted_f1 - zero_summary.weighted_f1,
            "kappa": few_summary.kappa - zero_summary.kappa,
        },
        "openai": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_base": api_base,
        },
        "max_examples_per_label": args.max_examples_per_label,
        "fewshot_source": str(args.fewshot),
        "taxonomy_source": str(args.taxonomy),
        "dataset_source": str(args.dataset),
    }
    meta_path.write_text(json.dumps(delta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved run meta: {meta_path}")

    print("\n=== Experiment 2 Results ===")
    print(f"N={delta['n']}")
    print("Zero-shot:", zero_summary)
    print("Few-shot :", few_summary)
    print("Diff (Few - Zero):", delta["diff_few_minus_zero"])


if __name__ == "__main__":
    main()