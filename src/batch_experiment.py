"""batch_experiment.py

論文 第5章「実験結果」用のバッチ実験スクリプト。

目的:
- 56件（正例28/負例28）の広告データに対して、提案手法（Few-shot + Graph RAG）相当の処理フローを一括実行
- 各広告で「リスク有無（0/1）」を推論し、正解ラベル(Tag)と比較して Accuracy / Precision / Recall / F1 / PR-AUC を算出
- 1件ごとの推論結果（予測、スコア、根拠パス等）をCSV/JSONLとして保存

前提:
- DATA_FILE はCSVで、少なくとも以下の列を持つこと（論文 表3）:
  - ID: 事例ID
  - Tag: 正例(1) / 負例(0)
  - Label: 炎上分類(1/2/3) ※本スクリプトでは集計用に保持（2値判定の正解はTag）
  - Text: 広告コピー本文

このスクリプトが依存するモジュール（プロジェクト内に存在する想定）:
- processor.py : 広告テキストから fact extraction（構造化抽出）
- loader.py    : Neo4jへ広告インスタンス投入
- mapper.py    : Association -> Concept のマッピング
- rag_app.py   : Graph RAG相当のクエリでリスクパス抽出

実行コマンドの例（ベースライン手法を実施する場合）
python3 batch_experiment.py --method proposed --era 2020s

※これらのAPIが未整備でも、関数名を合わせて実装すれば本スクリプトはそのまま動作します。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure this script's directory (src/) is on sys.path so that `import baselines` works
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from neo4j import GraphDatabase

from config import DATA_FILE, NEO4J_AUTH, NEO4J_URI


# --- Optional imports (project modules) ---
# 依存モジュールの関数シグネチャはこのスクリプト側で期待値を明示する。
try:
    import processor  # type: ignore
except Exception:  # pragma: no cover
    processor = None

try:
    import loader  # type: ignore
except Exception:  # pragma: no cover
    loader = None

try:
    import mapper  # type: ignore
except Exception:  # pragma: no cover
    mapper = None

try:
    import rag_app  # type: ignore
except Exception:  # pragma: no cover
    rag_app = None

_BASELINES_IMPORT_ERROR: Optional[Exception] = None
try:
    import baselines  # type: ignore
except Exception as e:  # pragma: no cover
    baselines = None
    _BASELINES_IMPORT_ERROR = e


# --- Metrics ---
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        precision_recall_curve,
        auc,
    )
except Exception:  # pragma: no cover
    accuracy_score = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    precision_recall_curve = None
    auc = None


@dataclass
class ExperimentRow:
    ad_id: str
    tag_true: int
    label: Optional[int]
    text: str
    meta: Dict[str, Any]


@dataclass
class PredictionResult:
    ad_id: str
    method: str
    y_true: int
    y_pred: int
    risk_score: float
    paths: List[Dict[str, Any]]
    facts: Dict[str, Any]
    mapped: Dict[str, Any]
    error: Optional[str] = None


def _setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _require_modules() -> None:
    missing = []
    if processor is None:
        missing.append("processor.py")
    if loader is None:
        missing.append("loader.py")
    if mapper is None:
        missing.append("mapper.py")
    if rag_app is None:
        missing.append("rag_app.py")
    if missing:
        raise RuntimeError(
            "必要モジュールがimportできません: "
            + ", ".join(missing)
            + "\n"
            + "これらがsrc配下に存在し、Pythonのimportパスに含まれていることを確認してください。"
        )


def _require_baselines() -> None:
    if baselines is None:
        detail = f" / cause: {_BASELINES_IMPORT_ERROR}" if _BASELINES_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "baselines.py がimportできません。src/baselines.py が存在し、Pythonのimportパスに含まれていることを確認してください。" + detail
        )


def _require_metrics() -> None:
    if any(
        x is None
        for x in [
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
            precision_recall_curve,
            auc,
        ]
    ):
        raise RuntimeError(
            "評価指標の算出に scikit-learn が必要です。`pip install scikit-learn` を実行してください。"
        )


def _resolve_default_data_path() -> Path:
    """既定のデータパスを解決する。

    優先順位:
    1) リポジトリ構造を前提とした ../data/experiment_data 配下のCSV（research_data.csv があれば優先）
    2) config.DATA_FILE
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # master_experiment/
    cand_dir = repo_root / "data" / "experiment_data"
    if cand_dir.exists():
        # Prefer a canonical name if present
        preferred = cand_dir / "research_data.csv"
        if preferred.exists():
            return preferred
        # Otherwise pick first csv
        csvs = sorted(cand_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
    return Path(DATA_FILE)


def load_dataset(path: Path, limit: Optional[int] = None) -> List[ExperimentRow]:
    """実験データCSVを読み込む。

    想定する列名（どちらでも可）:
      - ID / id: 事例ID
      - Tag / tag: 正例(1) / 負例(0)
      - Text / text / copy_candidates: 広告コピー本文
      - Label / label: 炎上分類(1/2/3) ※任意

    それ以外の列は meta として各行に保持する（後で分析用に保存可能）。
    """
    if not path.exists():
        raise FileNotFoundError(f"DATA_FILE not found: {path}")

    df = pd.read_csv(path)

    # Column normalization (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}

    def _pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    id_col = _pick("ID", "id")
    tag_col = _pick("Tag", "tag")
    text_col = _pick("Text", "text", "copy_candidates", "copy", "ad_text")

    if id_col is None or tag_col is None or text_col is None:
        raise ValueError(
            "CSVに必要列がありません。必要: (ID/id), (Tag/tag), (Text/text/copy_candidates). "
            f"found={list(df.columns)}"
        )

    label_col = _pick("Label", "label")

    if limit is not None:
        df = df.head(limit)

    rows: List[ExperimentRow] = []
    for _, r in df.iterrows():
        ad_id = str(r[id_col])
        tag_true = int(r[tag_col])
        text = str(r[text_col]) if not pd.isna(r[text_col]) else ""

        label_val: Optional[int] = None
        if label_col is not None and not pd.isna(r[label_col]):
            try:
                label_val = int(r[label_col])
            except Exception:
                label_val = None

        # keep other columns as meta
        meta: Dict[str, Any] = {}
        for c in df.columns:
            if c in {id_col, tag_col, text_col}:
                continue
            if label_col is not None and c == label_col:
                continue
            v = r[c]
            if pd.isna(v):
                continue
            meta[c] = v

        rows.append(
            ExperimentRow(
                ad_id=ad_id,
                tag_true=tag_true,
                label=label_val,
                text=text,
                meta=meta,
            )
        )

    return rows


def run_one(
    driver: Any,
    row: ExperimentRow,
    *,
    method: str,
    overwrite: bool,
    association_top_k: int,
    concept_similarity_threshold: float,
    max_paths: int,
    era: str,
) -> PredictionResult:
    """1広告分の一連処理。

    期待する依存モジュールAPI（例）:

    processor.extract_facts(text: str, ad_id: str | None = None) -> dict
      - Expression / Evidence / Role / Association 等を含む構造化データ

    loader.upsert_ad_instance(driver, ad_id: str, text: str, tag: int, label: int|None, facts: dict, overwrite: bool) -> dict
      - Neo4jへ広告インスタンス投入（ノード/リレーション作成）

    mapper.map_associations_to_concepts(driver, ad_id: str, top_k: int, similarity_threshold: float, overwrite: bool) -> dict
      - Association -> Concept のマッピング結果（何件接続したか等）

    rag_app.extract_risk_paths(driver, ad_id: str, max_paths: int) -> dict
      - {"risk_score": float, "paths": [ ... ]}
      - pathsは推論パス（起点Expression/AssociationからRiskFactor/Normへ）を表現するdictの配列

    判定ルール（2値）:
    - y_pred = 1 if risk_score > 0 または paths が1本以上存在
    """

    try:
        # --- Baselines ---
        if method in {"zero-shot", "few-shot", "text-rag"}:
            _require_baselines()

            if method == "zero-shot":
                y_pred = int(baselines.predict_zero_shot(row.text))  # type: ignore
            elif method == "few-shot":
                y_pred = int(baselines.predict_few_shot(row.text))  # type: ignore
            else:  # text-rag
                y_pred = int(baselines.predict_text_rag(row.text, driver))  # type: ignore

            # baselineは0/1をそのままスコアとして扱う
            return PredictionResult(
                ad_id=row.ad_id,
                method=method,
                y_true=row.tag_true,
                y_pred=y_pred,
                risk_score=float(y_pred),
                paths=[],
                facts={},
                mapped={},
                error=None,
            )

        facts: Dict[str, Any] = processor.extract_facts(row.text, ad_id=row.ad_id, meta=row.meta)  # type: ignore

        loaded: Dict[str, Any] = loader.upsert_ad_instance(  # type: ignore
            driver,
            ad_id=row.ad_id,
            text=row.text,
            tag=row.tag_true,
            label=row.label,
            facts=facts,
            overwrite=overwrite,
        )

        mapped: Dict[str, Any] = mapper.map_associations_to_concepts(  # type: ignore
            driver,
            ad_id=row.ad_id,
            top_k=association_top_k,
            similarity_threshold=concept_similarity_threshold,
            overwrite=overwrite,
        )

        rag_out: Dict[str, Any] = rag_app.extract_risk_paths(  # type: ignore
            driver,
            ad_id=row.ad_id,
            max_paths=max_paths,
            era=era,
        )

        risk_score = float(rag_out.get("risk_score", 0.0))
        paths = rag_out.get("paths", [])
        if paths is None:
            paths = []

        # 2値判定ルール（論文の「明示的な推論パスが存在する場合にのみリスク検出」に整合）
        y_pred = 1 if (len(paths) > 0 or risk_score > 0.0) else 0

        return PredictionResult(
            ad_id=row.ad_id,
            method=method,
            y_true=row.tag_true,
            y_pred=y_pred,
            risk_score=risk_score,
            paths=list(paths),
            facts=facts,
            mapped=mapped,
            error=None,
        )

    except Exception as e:
        logging.exception("Failed on ad_id=%s", row.ad_id)
        return PredictionResult(
            ad_id=row.ad_id,
            method=method,
            y_true=row.tag_true,
            y_pred=0,
            risk_score=0.0,
            paths=[],
            facts={},
            mapped={},
            error=str(e),
        )


def compute_metrics(results: List[PredictionResult]) -> Dict[str, Any]:
    _require_metrics()

    y_true = [r.y_true for r in results]
    y_pred = [r.y_pred for r in results]

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # PR-AUCはスコアが必要。ここでは risk_score を使用（未実装の場合は0になり得る）
    scores = [float(r.risk_score) for r in results]
    pr_auc_val: Optional[float] = None
    try:
        p, r, _ = precision_recall_curve(y_true, scores)
        pr_auc_val = float(auc(r, p))
    except Exception:
        pr_auc_val = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "n": len(results),
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": pr_auc_val,
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def save_outputs(
    out_dir: Path,
    results: List[PredictionResult],
    metrics: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-ad CSV
    df = pd.DataFrame(
        [
            {
                "ad_id": r.ad_id,
                "method": r.method,
                "y_true": r.y_true,
                "y_pred": r.y_pred,
                "risk_score": r.risk_score,
                "n_paths": len(r.paths),
                "error": r.error,
            }
            for r in results
        ]
    )
    df.to_csv(out_dir / "predictions.csv", index=False)

    # 2) jsonl for detailed artifacts (facts/paths)
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "ad_id": r.ad_id,
                        "method": r.method,
                        "y_true": r.y_true,
                        "y_pred": r.y_pred,
                        "risk_score": r.risk_score,
                        "paths": r.paths,
                        "facts": r.facts,
                        "mapped": r.mapped,
                        "error": r.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # 3) metrics
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def parse_args(default_data: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch experiment for Graph RAG ad backlash risk detection")

    p.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Path to CSV dataset (default: ../data/experiment_data/*.csv if exists, else config.DATA_FILE)",
    )
    p.add_argument("--out", type=str, default=None, help="Output directory (default: ./outputs/<timestamp>)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows (debug)")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    # Experiment params
    p.add_argument("--era", type=str, default="2020s", choices=["2020s", "2010s"], help="Era for risk judgement")

    # Method selection (for baselines / ablations)
    p.add_argument(
        "--method",
        type=str,
        default="proposed",
        choices=["proposed", "zero-shot", "few-shot", "text-rag"],
        help="Method to evaluate: proposed (default), zero-shot, few-shot, text-rag",
    )

    # Control flags
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing ad instances / mappings in Neo4j")

    # Mapper params
    p.add_argument("--association-top-k", type=int, default=8, help="Top-k candidates for Association->Concept mapping")
    p.add_argument(
        "--concept-similarity-threshold",
        type=float,
        default=0.70,
        help="Similarity threshold for accepting Association->Concept edges",
    )

    # RAG params
    p.add_argument("--max-paths", type=int, default=20, help="Max number of risk paths to return per ad")

    return p.parse_args()


def main() -> None:
    default_data = _resolve_default_data_path()
    args = parse_args(default_data)
    _setup_logger(args.log_level)

    # Require only what is needed for the chosen method
    if args.method == "proposed":
        _require_modules()
    else:
        _require_baselines()

    data_path = Path(args.data)
    rows = load_dataset(data_path, limit=args.limit)
    logging.info("Loaded %d records from %s", len(rows), data_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("outputs") / ts

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    results: List[PredictionResult] = []
    try:
        for i, row in enumerate(rows, start=1):
            logging.info("[%d/%d] Running ad_id=%s", i, len(rows), row.ad_id)
            res = run_one(
                driver,
                row,
                method=str(args.method),
                overwrite=bool(args.overwrite),
                association_top_k=int(args.association_top_k),
                concept_similarity_threshold=float(args.concept_similarity_threshold),
                max_paths=int(args.max_paths),
                era=str(args.era),
            )
            results.append(res)

        metrics = compute_metrics(results)
        save_outputs(out_dir, results, metrics)

        logging.info("=== Summary ===")
        logging.info("n=%d", metrics.get("n"))
        logging.info("Accuracy=%.4f", metrics.get("accuracy"))
        logging.info("Precision=%.4f | Recall=%.4f | F1=%.4f", metrics.get("precision"), metrics.get("recall"), metrics.get("f1"))
        if metrics.get("pr_auc") is not None:
            logging.info("PR-AUC=%.4f", metrics.get("pr_auc"))
        logging.info("Confusion: %s", metrics.get("confusion"))
        logging.info("Outputs saved to: %s", out_dir)

    finally:
        driver.close()


if __name__ == "__main__":
    main()