from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RunConfig


def _rolling_shift(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean().shift(1).fillna(0)


def build_feature_frame(tables: Dict[str, pd.DataFrame], cfg: RunConfig) -> pd.DataFrame:
    pr = tables["pull_request"].copy()
    pr["merged_at"] = pd.to_datetime(pr["merged_at"], errors="coerce")
    pr["created_at"] = pd.to_datetime(pr["created_at"], errors="coerce")
    pr["id"] = pd.to_numeric(pr["id"], errors="coerce")
    pr["repo_id"] = pd.to_numeric(pr["repo_id"], errors="coerce")
    pr["user_id"] = pd.to_numeric(pr["user_id"], errors="coerce")
    pr["is_merged"] = pr["merged_at"].notna().astype(int)
    pr = pr.sort_values([cfg.repo_col, "created_at"]).reset_index(drop=True)

    churn = (
        tables["pr_commit_details"]
        .groupby("pr_id")
        .agg(
            total_additions=("commit_stats_additions", "sum"),
            total_deletions=("commit_stats_deletions", "sum"),
            total_changes=("commit_stats_total", "sum"),
            num_files=("filename", "nunique"),
        )
        .reset_index()
        .rename(columns={"pr_id": "id"})
    )
    churn["id"] = pd.to_numeric(churn["id"], errors="coerce")
    pr = pr.merge(churn, on="id", how="left")
    for col in ["total_additions", "total_deletions", "total_changes", "num_files"]:
        pr[col] = pr[col].fillna(0)

    pr["log_additions"] = np.log1p(pr["total_additions"])
    pr["log_deletions"] = np.log1p(pr["total_deletions"])
    pr["log_changes"] = np.log1p(pr["total_changes"])
    pr["log_num_files"] = np.log1p(pr["num_files"])
    pr["complexity_score"] = pr["log_changes"] * np.log1p(pr["num_files"])
    pr["delta_complexity"] = pr.groupby(cfg.repo_col)["complexity_score"].diff().fillna(0)

    commit_count = (
        tables["pr_commits"]
        .groupby("pr_id")
        .size()
        .reset_index(name="commit_count")
        .rename(columns={"pr_id": "id"})
    )
    commit_count["id"] = pd.to_numeric(commit_count["id"], errors="coerce")
    pr = pr.merge(commit_count, on="id", how="left")
    pr["commit_count"] = pr["commit_count"].fillna(0)

    pr["prev_is_merged"] = pr.groupby(cfg.repo_col)["is_merged"].shift(1).fillna(0)
    pr["roll_merge_rate"] = pr.groupby(cfg.repo_col)["is_merged"].transform(
        lambda s: _rolling_shift(s, cfg.rolling_window)
    )

    task_map = tables["pr_task_type"][["id", "type"]].rename(columns={"type": "task_type"}).copy()
    task_map["id"] = pd.to_numeric(task_map["id"], errors="coerce")
    pr = pr.merge(task_map, on="id", how="left")
    pr["task_type"] = pr["task_type"].fillna("unknown")
    task_dummies = pd.get_dummies(pr["task_type"], prefix="task").astype(int)
    pr = pd.concat([pr, task_dummies], axis=1)

    repo = tables["repository"][["id", "stars", "forks"]].copy().rename(columns={"id": "repo_id"})
    repo["repo_id"] = pd.to_numeric(repo["repo_id"], errors="coerce")
    pr = pr.merge(repo, on="repo_id", how="left")
    pr["repo_log_stars"] = np.log1p(pr["stars"].fillna(0))
    pr["repo_log_forks"] = np.log1p(pr["forks"].fillna(0))

    agent_rates = pr.groupby("agent")["is_merged"].mean().rename("agent_global_rate")
    pr = pr.merge(agent_rates, on="agent", how="left")
    agent_task_rates = (
        pr.groupby(["agent", "task_type"])["is_merged"]
        .mean()
        .rename("agent_task_rate")
        .reset_index()
    )
    pr = pr.merge(agent_task_rates, on=["agent", "task_type"], how="left")
    pr["agent_task_rate"] = pr["agent_task_rate"].fillna(pr["agent_global_rate"])

    user = tables["user"][["id", "followers", "following"]].copy().rename(columns={"id": "user_id"})
    user["user_id"] = pd.to_numeric(user["user_id"], errors="coerce")
    pr = pr.merge(user, on="user_id", how="left")
    pr["user_followers"] = pr["followers"].fillna(0)
    pr["user_following"] = pr["following"].fillna(0)
    pr["follower_ratio"] = pr["user_followers"] / (pr["user_following"] + 1)

    pr["title_len"] = pr["title"].fillna("").str.len()
    pr["title_words"] = pr["title"].fillna("").str.split().str.len()
    pr["body_len"] = pr["body"].fillna("").str.len()
    pr["has_body"] = pr["body"].notna().astype(int)
    pr["has_emoji"] = (
        (pr["title"].fillna("") + pr["body"].fillna(""))
        .str.contains(r"[\U0001F300-\U0001FAFF]", regex=True)
        .astype(int)
    )

    related = tables["related_issue"].copy()
    related["pr_id"] = pd.to_numeric(related["pr_id"], errors="coerce")
    linked = related.groupby("pr_id").size().reset_index(name="linked_issue_count").rename(columns={"pr_id": "id"})
    pr = pr.merge(linked, on="id", how="left")
    pr["linked_issue_count"] = pr["linked_issue_count"].fillna(0)
    pr["has_linked_issue"] = (pr["linked_issue_count"] > 0).astype(int)

    reviews = tables["pr_reviews"].copy()
    reviews["pr_id"] = pd.to_numeric(reviews["pr_id"], errors="coerce")
    reviewer_count = (
        reviews.groupby("pr_id")["user"]
        .nunique()
        .reset_index(name="reviewer_count_current")
        .rename(columns={"pr_id": "id"})
    )
    pr = pr.merge(reviewer_count, on="id", how="left")
    pr["reviewer_count_current"] = pr["reviewer_count_current"].fillna(0)
    pr["prev_reviewer_cnt"] = pr.groupby(cfg.repo_col)["reviewer_count_current"].shift(1).fillna(0)

    timeline = tables["pr_timeline"].copy()
    timeline["pr_id"] = pd.to_numeric(timeline["pr_id"], errors="coerce")
    review_requested = (
        timeline[timeline["event"] == "review_requested"]
        .groupby("pr_id")
        .size()
        .reset_index(name="event_review_requested_current")
        .rename(columns={"pr_id": "id"})
    )
    pr = pr.merge(review_requested, on="id", how="left")
    pr["event_review_requested_current"] = pr["event_review_requested_current"].fillna(0)
    pr["event_review_requested"] = (
        pr.groupby(cfg.repo_col)["event_review_requested_current"].shift(1).fillna(0)
    )

    pr["day_of_week"] = pr["created_at"].dt.dayofweek
    pr["hour_of_day"] = pr["created_at"].dt.hour
    pr["is_weekend"] = (pr["day_of_week"] >= 5).astype(int)

    approved = (
        reviews[reviews["state"] == "APPROVED"]
        .groupby("pr_id")
        .size()
        .reset_index(name="approved_count")
        .rename(columns={"pr_id": "id"})
    )
    pr = pr.merge(approved, on="id", how="left")
    pr["approved_count"] = pr["approved_count"].fillna(0)
    pr["has_approval"] = (pr["approved_count"] > 0).astype(int)
    pr["reviewer_count"] = pr["reviewer_count_current"]

    changes_requested = (
        reviews[reviews["state"] == "CHANGES_REQUESTED"]
        .groupby("pr_id")
        .size()
        .reset_index(name="changes_requested_count")
        .rename(columns={"pr_id": "id"})
    )
    pr = pr.merge(changes_requested, on="id", how="left")
    pr["changes_requested_count"] = pr["changes_requested_count"].fillna(0)
    pr["was_revised"] = (pr["changes_requested_count"] > 0).astype(int)

    comments = tables["pr_comments"].copy()
    comments["pr_id"] = pd.to_numeric(comments["pr_id"], errors="coerce")
    comment_count = comments.groupby("pr_id").size().reset_index(name="pr_comment_count").rename(columns={"pr_id": "id"})
    pr = pr.merge(comment_count, on="id", how="left")
    pr["pr_comment_count"] = pr["pr_comment_count"].fillna(0)

    return pr


def compute_roberta_sentiment(tables: Dict[str, pd.DataFrame], cfg: RunConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model_name = cfg.sentiment_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    frames = []
    for tname in ["pr_comments", "pr_review_comments_v2"]:
        frame = tables[tname].copy()
        if "body" in frame.columns and "pr_id" in frame.columns:
            frames.append(frame[["pr_id", "body"]].dropna(subset=["body"]))

    comments = pd.concat(frames, ignore_index=True)
    comments = comments[comments["body"].astype(str).str.strip() != ""].copy()
    comments["pr_id"] = pd.to_numeric(comments["pr_id"], errors="coerce")

    unique = comments[["body"]].drop_duplicates().copy()
    texts = unique["body"].astype(str).tolist()
    scores = []
    for start in tqdm(range(0, len(texts), cfg.sentiment_batch_size), desc="RoBERTa scoring", unit="batch"):
        batch = texts[start : start + cfg.sentiment_batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.sentiment_max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        polarity = probs[:, 2] - probs[:, 0]
        scores.extend(polarity.tolist())

    unique["sent_rb"] = scores
    comments = comments.merge(unique, on="body", how="left")

    sentiment = (
        comments.groupby("pr_id")["sent_rb"]
        .mean()
        .reset_index()
        .rename(columns={"pr_id": "id"})
    )
    sentiment["id"] = pd.to_numeric(sentiment["id"], errors="coerce")

    metadata = {
        "raw_comment_rows": int(len(comments)),
        "unique_comment_texts": int(len(unique)),
        "prs_with_sentiment": int(sentiment["id"].nunique()),
        "mean_polarity": float(sentiment["sent_rb"].mean()),
    }
    return sentiment, metadata


def attach_sentiment(pr: pd.DataFrame, sentiment: pd.DataFrame, cfg: RunConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pr = pr.merge(sentiment, on="id", how="left")
    pr["sent_rb"] = pr["sent_rb"].fillna(0)
    pr["prev_sentiment"] = pr.groupby(cfg.repo_col)["sent_rb"].shift(1).fillna(0)
    pr["roll_sentiment"] = pr.groupby(cfg.repo_col)["sent_rb"].transform(
        lambda s: _rolling_shift(s, cfg.rolling_window)
    )
    coverage = float((pr["sent_rb"] != 0).mean())
    return pr, {"sentiment_coverage_nonzero": coverage}


def get_feature_sets(pr: pd.DataFrame) -> Dict[str, Tuple[list[str], str]]:
    task_cols = sorted([c for c in pr.columns if c.startswith("task_")])
    complexity = ["complexity_score"]
    history = ["roll_merge_rate", "prev_is_merged"]
    sentiment = ["prev_sentiment", "roll_sentiment"]
    real_churn = [
        "log_additions",
        "log_deletions",
        "log_changes",
        "log_num_files",
        "delta_complexity",
        "commit_count",
    ]
    agent_repo = ["agent_global_rate", "agent_task_rate", "repo_log_stars", "repo_log_forks"]
    context_user = [
        "user_followers",
        "follower_ratio",
        "has_linked_issue",
        "title_len",
        "body_len",
        "title_words",
        "has_body",
        "has_emoji",
        "prev_reviewer_cnt",
        "day_of_week",
        "hour_of_day",
        "is_weekend",
        "event_review_requested",
    ]
    oracle = ["approved_count", "has_approval", "reviewer_count", "was_revised", "pr_comment_count"]
    return {
        "A_baseline_time": (complexity, "time"),
        "B_history_time": (complexity + history, "time"),
        "C_sentiment_time": (complexity + history + ["prev_sentiment", "delta_complexity"], "time"),
        "D_real_churn_time": (real_churn + complexity + history + sentiment + task_cols, "time"),
        "E_agent_repo_agst": (real_churn + complexity + history + sentiment + agent_repo + task_cols, "agst"),
        "F_context_user_agst": (real_churn + complexity + history + sentiment + agent_repo + context_user + task_cols, "agst"),
        "G_oracle_agst": (real_churn + complexity + history + sentiment + agent_repo + context_user + oracle + task_cols, "agst"),
    }
