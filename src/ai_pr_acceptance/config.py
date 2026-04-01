from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RunConfig:
    hf_root: str = "hf://datasets/hao-li/AIDev/"
    output_dir: str = "outputs/run"
    repo_col: str = "repo_id"
    rolling_window: int = 3
    shap_sample: int = 5000
    seed: int = 42

    gb_n_estimators: int = 300
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.05
    gb_subsample: float = 0.8
    gb_min_samples_leaf: int = 10

    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_eval_metric: str = "logloss"

    sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"
    sentiment_max_length: int = 128
    sentiment_batch_size: int = 64

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self):
        return asdict(self)
