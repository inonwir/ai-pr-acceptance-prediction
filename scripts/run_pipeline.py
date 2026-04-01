from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_pr_acceptance.config import RunConfig
from ai_pr_acceptance.data import load_tables
from ai_pr_acceptance.features import attach_sentiment, build_feature_frame, compute_roberta_sentiment, get_feature_sets
from ai_pr_acceptance.figures import save_metrics_bar, save_shap_outputs
from ai_pr_acceptance.models import baseline_comparison, feature_importance, per_agent_auc, run_cross_validation
from ai_pr_acceptance.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI PR acceptance reproducibility pipeline.")
    parser.add_argument("--hf-root", default="hf://datasets/hao-li/AIDev/")
    parser.add_argument("--output-dir", default="outputs/run")
    parser.add_argument("--shap-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(hf_root=args.hf_root, output_dir=args.output_dir, shap_sample=args.shap_sample, seed=args.seed)
    output_dir = cfg.output_path()
    ensure_dir(output_dir)

    tables = load_tables(cfg.hf_root)
    pr = build_feature_frame(tables, cfg)
    sentiment_df, sentiment_meta = compute_roberta_sentiment(tables, cfg)
    pr, sentiment_attach_meta = attach_sentiment(pr, sentiment_df, cfg)

    feature_sets = get_feature_sets(pr)
    data = pr.copy()
    results = run_cross_validation(data, feature_sets, cfg)

    metrics_rows = []
    for model_name, res in results.items():
        metrics_rows.append(
            {
                "model": model_name,
                "cv_type": res["cv_type"],
                "gb_auc_mean": res["gb_mean"],
                "gb_auc_std": res["gb_std"],
                "xgb_auc_mean": res["xgb_mean"],
                "xgb_auc_std": res["xgb_std"],
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics_table4.csv", index=False)

    baselines = baseline_comparison(data, cfg)
    baselines.to_csv(output_dir / "baseline_comparison.csv", index=False)

    for model_name, file_name in [
        ("D_real_churn_time", "feature_importance_D.csv"),
        ("F_context_user_agst", "feature_importance_F.csv"),
        ("G_oracle_agst", "feature_importance_G.csv"),
    ]:
        importance_df, fitted_model = feature_importance(data, results, model_name, cfg)
        importance_df.to_csv(output_dir / file_name, index=False)
        if model_name in {"D_real_churn_time", "F_context_user_agst"}:
            shap_features = results[model_name]["features"]
            X = data[shap_features].fillna(0).sample(n=min(cfg.shap_sample, len(data)), random_state=cfg.seed)
            save_shap_outputs(fitted_model, X, output_dir / model_name)

    per_agent = per_agent_auc(data, results, "D_real_churn_time", cfg)
    per_agent.to_csv(output_dir / "per_agent_auc_model_d.csv", index=False)

    save_metrics_bar(metrics, output_dir / "fig_model_auc.png")

    metadata = {
        "config": cfg.to_dict(),
        "n_pull_requests": int(len(pr)),
        "merge_rate": float(pr["is_merged"].mean()),
        "n_repositories": int(pr["repo_id"].nunique()),
        "n_agents": int(pr["agent"].nunique()),
        "sentiment": {**sentiment_meta, **sentiment_attach_meta},
    }
    write_json(output_dir / "run_metadata.json", metadata)
    print(f"Done. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
