from __future__ import annotations

from typing import Dict

import pandas as pd


TABLE_NAMES = [
    "pull_request",
    "pr_comments",
    "pr_review_comments_v2",
    "pr_commit_details",
    "pr_commits",
    "pr_task_type",
    "repository",
    "user",
    "pr_reviews",
    "pr_timeline",
    "related_issue",
]


def load_tables(hf_root: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        tables[name] = pd.read_parquet(f"{hf_root}{name}.parquet")
    return tables
