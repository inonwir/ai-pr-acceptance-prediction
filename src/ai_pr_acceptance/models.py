from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from .config import RunConfig


def _gb_model(cfg: RunConfig) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=cfg.gb_n_estimators,
        max_depth=cfg.gb_max_depth,
        learning_rate=cfg.gb_learning_rate,
        subsample=cfg.gb_subsample,
        min_samples_leaf=cfg.gb_min_samples_leaf,
        random_state=cfg.seed,
    )


def _xgb_model(cfg: RunConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        eval_metric=cfg.xgb_eval_metric,
        random_state=cfg.seed,
    )


def run_cross_validation(
    data: pd.DataFrame,
    feature_sets: Dict[str, Tuple[List[str], str]],
    cfg: RunConfig,
) -> Dict[str, dict]:
    y = data["is_merged"]
    groups = data["agent"]
    cv_time = StratifiedKFold(n_splits=5, shuffle=False)
    cv_agst = StratifiedGroupKFold(n_splits=5)

    results: Dict[str, dict] = {}
    for name, (features, cv_type) in feature_sets.items():
        valid_features = [f for f in features if f in data.columns]
        X = data[valid_features].fillna(0)
        if cv_type == "time":
            gb_scores = cross_val_score(_gb_model(cfg), X, y, cv=cv_time, scoring="roc_auc")
            xgb_scores = cross_val_score(_xgb_model(cfg), X, y, cv=cv_time, scoring="roc_auc")
        else:
            gb_scores = cross_val_score(_gb_model(cfg), X, y, cv=cv_agst, scoring="roc_auc", groups=groups)
            xgb_scores = None
        results[name] = {
            "features": valid_features,
            "cv_type": cv_type,
            "gb_scores": gb_scores,
            "xgb_scores": xgb_scores,
            "gb_mean": float(np.mean(gb_scores)),
            "gb_std": float(np.std(gb_scores)),
            "xgb_mean": float(np.mean(xgb_scores)) if xgb_scores is not None else None,
            "xgb_std": float(np.std(xgb_scores)) if xgb_scores is not None else None,
        }
    return results


def baseline_comparison(data: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    y = data["is_merged"]
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    baselines = {
        "majority_class": (DummyClassifier(strategy="most_frequent"), ["complexity_score"]),
        "logistic_complexity": (
            LogisticRegression(max_iter=1000, random_state=cfg.seed),
            ["log_changes", "log_num_files", "complexity_score"],
        ),
        "logistic_with_roll_merge_rate": (
            LogisticRegression(max_iter=1000, random_state=cfg.seed),
            ["log_changes", "log_num_files", "complexity_score", "roll_merge_rate"],
        ),
    }
    rows = []
    for name, (clf, feats) in baselines.items():
        scores = cross_val_score(clf, data[feats].fillna(0), y, cv=cv, scoring="roc_auc")
        rows.append({"model": name, "auc_mean": float(scores.mean()), "auc_std": float(scores.std())})
    return pd.DataFrame(rows)


def feature_importance(data: pd.DataFrame, results: Dict[str, dict], model_name: str, cfg: RunConfig) -> Tuple[pd.DataFrame, GradientBoostingClassifier]:
    features = results[model_name]["features"]
    X = data[features].fillna(0)
    y = data["is_merged"]
    model = _gb_model(cfg)
    model.fit(X, y)
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_ * 100,
        }
    ).sort_values("importance", ascending=False)
    return importance, model


def per_agent_auc(data: pd.DataFrame, results: Dict[str, dict], model_name: str, cfg: RunConfig) -> pd.DataFrame:
    features = results[model_name]["features"]
    X = data[features].fillna(0)
    y = data["is_merged"]
    model = _gb_model(cfg)
    model.fit(X, y)
    rows = []
    for agent in sorted(data["agent"].dropna().unique()):
        mask = data["agent"] == agent
        y_agent = y[mask]
        if y_agent.nunique() < 2:
            continue
        auc = roc_auc_score(y_agent, model.predict_proba(X[mask])[:, 1])
        rows.append(
            {
                "agent": agent,
                "auc": float(auc),
                "merge_rate": float(y_agent.mean()),
                "n_prs": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("agent")
