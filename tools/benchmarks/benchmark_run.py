"""
DermaMind.ai Benchmarking Script
Measures execution time and memory usage for each phase of the discovery pipeline.
Outputs a structured JSON report to data/benchmark_results_<timestamp>.json.

Usage:
    .venv\\Scripts\\python.exe tools/benchmarks/benchmark_run.py
    .venv\\Scripts\\python.exe tools/benchmarks/benchmark_run.py --cohort TCGA-BRCA --seed 123 --n_hypotheses 3
"""

import os
import sys
import json
import time
import argparse
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _measure(label: str, fn, *args, **kwargs) -> Dict[str, Any]:
    """Run fn(*args, **kwargs), measuring wall time and peak memory. Returns result + metrics."""
    tracemalloc.start()
    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "phase": label,
        "wall_time_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak_bytes / 1024 / 1024, 3),
        "success": error is None,
        "error": error,
        "_result": result,
    }


def run_benchmark(cohort_id: str = "TCGA-SKCM",
                  random_seed: int = 42,
                  n_hypotheses: int = 2) -> Dict[str, Any]:
    """
    Execute each pipeline phase individually and record timing + memory metrics.

    Phases measured:
      1. Data loading (GDC / local mirror)
      2. Pathology feature extraction (mock mode)
      3. Knowledge fabric construction
      4. KM survival analysis (DSL template)
      5. Cox regression (DSL template)
      6. ML survival prediction (DSL template)
      7. KG constrained walk
      8. KG confidence scoring
    """
    print(f"\n{'='*60}")
    print(f"  DermaMind.ai Benchmark  |  cohort={cohort_id}  seed={random_seed}")
    print(f"{'='*60}\n")

    phases = []
    meta = {
        "cohort_id": cohort_id,
        "random_seed": random_seed,
        "n_hypotheses": n_hypotheses,
        "timestamp": datetime.now().isoformat(),
        "schema_version": "1.0",
    }

    # ------------------------------------------------------------------ #
    # Phase 1: Data loading
    # ------------------------------------------------------------------ #
    print("[1/8] Data loading...")
    from tools.enhanced_data_client import get_data_client

    data_client = get_data_client()
    p1 = _measure(
        "data_loading",
        data_client.get_survival_analysis_ready_data,
        cohort_id,
        missing_data_strategy="impute_median",
    )
    df_clinical = p1.pop("_result")
    phases.append(p1)
    n_samples = len(df_clinical) if df_clinical is not None and not df_clinical.empty else 0
    p1["n_samples_loaded"] = n_samples
    print(f"   → {n_samples} samples | {p1['wall_time_seconds']:.3f}s | {p1['peak_memory_mb']:.1f} MB")

    if df_clinical is None or df_clinical.empty:
        print("   ⚠️  No clinical data returned; skipping downstream phases.")
        df_clinical = None

    # ------------------------------------------------------------------ #
    # Phase 2: Pathology feature extraction (mock mode)
    # ------------------------------------------------------------------ #
    print("[2/8] Pathology feature extraction (mock)...")
    from tools.enhanced_pathology import extract_features_for_dataset

    pathology_csv = f"data/benchmark_pathology_{cohort_id.lower()}.csv"
    case_ids = (
        df_clinical["submitter_id"].tolist()[:20]
        if df_clinical is not None and "submitter_id" in df_clinical.columns
        else [f"CASE-{i:04d}" for i in range(20)]
    )
    p2 = _measure(
        "pathology_feature_extraction",
        extract_features_for_dataset,
        wsi_dir="data/wsi_slides",
        output_csv_path=pathology_csv,
        case_ids=case_ids,
        mock_mode=True,
    )
    _ = p2.pop("_result")
    phases.append(p2)
    print(f"   → {len(case_ids)} cases | {p2['wall_time_seconds']:.3f}s | {p2['peak_memory_mb']:.1f} MB")

    # ------------------------------------------------------------------ #
    # Phase 3: Knowledge fabric construction
    # ------------------------------------------------------------------ #
    print("[3/8] Knowledge fabric construction...")
    from tools.knowledge_fabric import MedicalKnowledgeBuilder

    p3 = _measure("knowledge_fabric_build", MedicalKnowledgeBuilder.build_oncology_knowledge_base)
    kg = p3.pop("_result")
    phases.append(p3)
    kg_stats = kg.get_statistics() if kg is not None else {}
    p3["kg_nodes"] = kg_stats.get("total_nodes", 0)
    p3["kg_edges"] = kg_stats.get("total_edges", 0)
    print(f"   → {p3['kg_nodes']} nodes, {p3['kg_edges']} edges | {p3['wall_time_seconds']:.3f}s | {p3['peak_memory_mb']:.1f} MB")

    # ------------------------------------------------------------------ #
    # Phase 4–6: Statistical analyses via DSL (on synthetic data if no real data)
    # ------------------------------------------------------------------ #
    import numpy as np
    import pandas as pd

    if df_clinical is not None and len(df_clinical) >= 30:
        bench_df = df_clinical.copy()
        # Ensure a grouping column exists for KM
        if "stage_group" not in bench_df.columns:
            bench_df["stage_group"] = np.random.choice(["Low", "High"], size=len(bench_df))
    else:
        print("   ℹ️  Using synthetic data for stats phases (no/small real cohort).")
        np.random.seed(random_seed)
        n = 200
        bench_df = pd.DataFrame({
            "time": np.random.exponential(400, n) + 50,
            "event": np.random.binomial(1, 0.45, n),
            "stage_group": np.random.choice(["Low", "High"], n),
            "age": np.random.normal(58, 12, n),
            "gene_mutated": np.random.choice([0, 1], p=[0.75, 0.25], size=n),
        })

    from tools.enhanced_clinical_stats import EnhancedStatsEngine

    engine = EnhancedStatsEngine(random_seed=random_seed)

    # Phase 4: KM
    print("[4/8] Kaplan-Meier DSL analysis...")
    p4 = _measure(
        "kaplan_meier_dsl",
        engine.execute_analysis_dsl,
        bench_df,
        {"template": "kaplan_meier", "parameters": {"time_col": "time", "event_col": "event", "group_col": "stage_group"}},
    )
    km_res = p4.pop("_result")
    phases.append(p4)
    p4["p_value"] = km_res.get("p_value") if isinstance(km_res, dict) else None
    print(f"   → p={p4['p_value']} | {p4['wall_time_seconds']:.3f}s | {p4['peak_memory_mb']:.1f} MB")

    # Phase 5: Cox
    print("[5/8] Cox regression DSL analysis...")
    bench_df["stage_num"] = (bench_df["stage_group"] == "High").astype(int)
    p5 = _measure(
        "cox_regression_dsl",
        engine.execute_analysis_dsl,
        bench_df,
        {"template": "cox_regression", "parameters": {"time_col": "time", "event_col": "event", "predictor_cols": ["stage_num", "age"]}},
    )
    cox_res = p5.pop("_result")
    phases.append(p5)
    p5["hazard_ratio"] = cox_res.get("hazard_ratio") if isinstance(cox_res, dict) else None
    p5["c_index"] = cox_res.get("c_index") if isinstance(cox_res, dict) else None
    print(f"   → HR={p5['hazard_ratio']} CI={p5['c_index']} | {p5['wall_time_seconds']:.3f}s | {p5['peak_memory_mb']:.1f} MB")

    # Phase 6: ML survival
    print("[6/8] ML survival DSL analysis...")
    p6 = _measure(
        "ml_survival_dsl",
        engine.execute_analysis_dsl,
        bench_df,
        {"template": "ml_survival", "parameters": {"feature_cols": ["age", "gene_mutated", "stage_num"], "target_col": "event", "time_col": "time", "model_type": "random_forest"}},
    )
    ml_res = p6.pop("_result")
    phases.append(p6)
    p6["auc_score"] = ml_res.get("c_index") if isinstance(ml_res, dict) else None
    print(f"   → AUC={p6['auc_score']} | {p6['wall_time_seconds']:.3f}s | {p6['peak_memory_mb']:.1f} MB")

    # ------------------------------------------------------------------ #
    # Phase 7: KG constrained walk
    # ------------------------------------------------------------------ #
    print("[7/8] Knowledge graph constrained walk...")
    if kg is not None:

        def _walk():
            return kg.constrained_walk("BRAF", ["MUTATED_IN", "TREATS"])

        p7 = _measure("kg_constrained_walk", _walk)
        walk_res = p7.pop("_result")
        phases.append(p7)
        p7["paths_found"] = len(walk_res) if walk_res else 0
        print(f"   → {p7['paths_found']} paths | {p7['wall_time_seconds']:.3f}s | {p7['peak_memory_mb']:.1f} MB")
    else:
        phases.append({"phase": "kg_constrained_walk", "skipped": True})
        print("   → skipped (no KG)")

    # ------------------------------------------------------------------ #
    # Phase 8: KG confidence scoring
    # ------------------------------------------------------------------ #
    print("[8/8] Knowledge graph confidence scoring...")
    if kg is not None:
        sample_hyp = "BRAF mutation is associated with Melanoma and response to Dabrafenib treatment"

        def _score():
            return kg.calculate_hypothesis_prior_score(sample_hyp)

        p8 = _measure("kg_confidence_scoring", _score)
        score_res = p8.pop("_result")
        phases.append(p8)
        p8["confidence_score"] = score_res.get("score") if isinstance(score_res, dict) else None
        p8["evidence_path"] = score_res.get("evidence_path") if isinstance(score_res, dict) else None
        print(f"   → score={p8['confidence_score']} | {p8['wall_time_seconds']:.3f}s | {p8['peak_memory_mb']:.1f} MB")
    else:
        phases.append({"phase": "kg_confidence_scoring", "skipped": True})
        print("   → skipped (no KG)")

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    total_time = sum(p.get("wall_time_seconds", 0) for p in phases)
    total_memory_peak = max((p.get("peak_memory_mb", 0) for p in phases), default=0)
    n_failed = sum(1 for p in phases if not p.get("success", True) and not p.get("skipped"))

    summary = {
        "total_wall_time_seconds": round(total_time, 4),
        "peak_memory_mb_across_phases": round(total_memory_peak, 3),
        "n_phases": len(phases),
        "n_failed_phases": n_failed,
        "all_phases_passed": n_failed == 0,
    }

    report = {**meta, "summary": summary, "phases": phases}

    # ------------------------------------------------------------------ #
    # Save report
    # ------------------------------------------------------------------ #
    Path("data").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/benchmark_results_{cohort_id.lower()}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Benchmark complete")
    print(f"  Total time : {total_time:.3f}s")
    print(f"  Peak memory: {total_memory_peak:.1f} MB")
    print(f"  Phases OK  : {len(phases) - n_failed}/{len(phases)}")
    print(f"  Report     : {out_path}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DermaMind.ai pipeline benchmark")
    parser.add_argument("--cohort", type=str, default="TCGA-SKCM", help="TCGA cohort project ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_hypotheses", type=int, default=2, help="Hypotheses per cohort (reserved for future use)")
    args = parser.parse_args()

    run_benchmark(cohort_id=args.cohort, random_seed=args.seed, n_hypotheses=args.n_hypotheses)
