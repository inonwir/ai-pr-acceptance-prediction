# Repository Culture Dominates AI Pull Request Acceptance

Reproducibility package for the paper:

**Repository Culture Dominates AI Pull Request Acceptance: An Empirical Study**

This repository contains a cleaned, script-based version of the original Colab notebook used to build features, run Models A-G, export paper tables, and generate SHAP / summary figures.

## What this repo reproduces

- Feature engineering from the public AIDev tables
- RoBERTa sentiment scoring using `cardiffnlp/twitter-roberta-base-sentiment`
- Cross-validation for Models A-G
- Baseline comparisons
- Gini importance
- SHAP plots for selected models
- Paper-ready CSV outputs for tables and metrics

## Dataset source

This project uses the public **AIDev** dataset released by Li et al. The dataset paper and official repository are:

- AIDev paper: `05 2507.15003v1 DEV AI dataset.pdf` in the working materials for this project
- Official AIDev repository: `https://github.com/SAILResearch/AI_Teammates_in_SE3`

## Important note about the analysis subset

This project **does not use the original AIDev-pop subset definition from the dataset paper**. Instead, it builds a custom analysis subset from the public AIDev tables and the filtering / feature-engineering logic implemented in this repository.

In the paper, state clearly that:

> We use the public AIDev tables, but define our own analysis subset rather than reusing the original AIDev-pop subset reported by Li et al.

## Tables used

The cleaned pipeline expects these AIDev parquet tables:

- `pull_request.parquet`
- `pr_comments.parquet`
- `pr_review_comments_v2.parquet`
- `pr_commit_details.parquet`
- `pr_commits.parquet`
- `pr_task_type.parquet`
- `repository.parquet`
- `user.parquet`
- `pr_reviews.parquet`
- `pr_timeline.parquet`
- `related_issue.parquet`

## Expected joins

- `pull_request.repo_id` -> `repository.id`
- `pull_request.user_id` -> `user.id`
- `pr_reviews.pr_id` -> `pull_request.id`
- `pr_comments.pr_id` -> `pull_request.id`
- `pr_review_comments_v2.pull_request_review_id` is not used as a join key directly; the pipeline scores text and aggregates back to PR level
- `related_issue.pr_id` -> `pull_request.id`

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── ai_pr_acceptance/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── figures.py
│       └── utils.py
├── scripts/
│   ├── run_pipeline.py
│   └── make_readme_artifacts.py
├── docs/
│   └── paper_insert_text.md
└── outputs/
    └── expected/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python scripts/run_pipeline.py --output-dir outputs/run_01
```

Optional arguments:

```bash
python scripts/run_pipeline.py \
  --hf-root "hf://datasets/hao-li/AIDev/" \
  --output-dir outputs/run_01 \
  --shap-sample 5000 \
  --seed 42
```

## Main outputs

After a successful run, the pipeline writes:

- `metrics_table4.csv`
- `baseline_comparison.csv`
- `feature_importance_D.csv`
- `feature_importance_F.csv`
- `feature_importance_G.csv`
- `per_agent_auc_model_d.csv`
- `run_metadata.json`
- figure PNG files

## Reproducibility checklist

Before making the repository public, confirm that the numbers in the paper and the outputs in this repo come from the **same frozen run**.

Check especially:

1. sentiment coverage in the paper vs the final run metadata
2. whether SHAP is reported for Model F in the paper and also generated here
3. whether the exact Model E/F/G AUC values in the paper match `metrics_table4.csv`
4. whether the final paper uses the same feature definitions as this repository

## Suggested paper text

Use the text in `docs/paper_insert_text.md` for the reproducibility / code-availability section.

## Recommended citation / archive practice

For submission, create a tagged GitHub release and archive it with Zenodo, then cite the DOI in the paper.

## Known cleanup decisions relative to the original notebook

This repository intentionally removes:

- exploratory discovery cells
- Colab-only download calls
- session-dependent fix cells
- manual one-off printing logic
- mixed notebook state assumptions

The cleaned version is designed to be script-driven and reviewable.
