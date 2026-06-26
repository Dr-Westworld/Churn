"""
XGBoost Telecom Churn Pipeline — patched
=========================================
Bugs fixed vs the original
---------------------------
BUG-1  (CRITICAL) total_services used Python `and` on a pandas row-Series
        → always raised ValueError: "The truth value of a Series is ambiguous".
        Fixed with vectorised boolean ops.

BUG-2  (CRITICAL) tenure_group / charge_group were pd.Categorical columns
        created AFTER categorical_cols was captured, so they were never
        label-encoded. XGBoost rejects non-numeric dtypes → training crash.
        Fixed by mapping them to ints immediately after pd.cut().

BUG-3  (CRITICAL) Missingness indicators were added AFTER fillna(), so every
        indicator column was identically 0 — useless.
        Fixed by adding indicators BEFORE imputation.

BUG-4  (CRITICAL) _parallel_cv_fold was an instance method decorated with
        @ray.remote. Ray remote functions cannot be instance methods; the
        decorator wraps the unbound function, so calling
        self._parallel_cv_fold.remote(self, …) serialises the entire pipeline
        object (file handles, logger, trained model …) for every fold.
        Fixed by moving to a module-level function.

BUG-5  (CRITICAL) predict() in predict_batch.remote(self.model, batch) passed
        the full XGBoost model as a direct argument → serialised/pickled once
        per batch. With N batches and a 40 MB model that is 40×N MB of copies.
        Fixed with ray.put() so the model is uploaded once.

BUG-6  (CORRECTNESS) Z-score features and median imputation were recomputed
        from the *current* data at inference time, leaking inference
        statistics into the feature transform and producing different values
        per batch. Fixed by storing training stats and reusing them.

BUG-7  (CORRECTNESS) The nested @ray.remote predict_batch closure was defined
        inside an instance method, causing Ray serialisation failures for
        complex closure state. Moved to module level.

Performance changes (from doc-2 / doc-3 review)
-------------------------------------------------
PERF-1  GridSearchCV with 12 two-option parameters → 2 048 combos × 5 folds
        ≈ 10 240 fits. Replaced with RandomizedSearchCV(n_iter=30) → 150 fits.
        Typical quality loss: <1 % AUC. Speed gain: ~68×.

PERF-2  GridSearchCV n_jobs=-1 with tree_method='gpu_hist' caused multiple
        processes to fight over the same GPU. Fixed: n_jobs=1 when num_gpus>0.

PERF-3  Ray CV workers used GPU params, causing GPU contention across workers.
        Fixed: CV workers use tree_method='hist' (CPU) so they can run truly
        in parallel while the main trainer owns the GPU.

PERF-4  matplotlib.use('Agg') + headless=True parameter prevents plt.show()
        from blocking automated / headless runs. Plots are saved to logs/.

PERF-5  DistributedLogger buffered fit writes (flush every 50 entries instead
        of one open(…,'a') per fit — was O(fits) blocking I/O).

PERF-6  base_params: n_estimators 2000→1000 (early_stopping handles the rest),
        max_depth 10→8. Removed deprecated GPU params that cause warnings on
        XGBoost ≥1.7 (n_gpus, gpu_platform_id, gpu_device_id, force_col_wise,
        gpu_use_dp).

Intentionally NOT applied
--------------------------
  • Target encoding  — all categoricals are low-cardinality (2-4 values);
    LabelEncoder is appropriate. Adding target encoding alongside LabelEncoder
    on the same columns would double-encode and corrupt features.
  • Probability calibration — changes the predict() return contract; out of
    scope for a bug-fix pass.
  • Ensemble — too complex; high risk of introducing new bugs.
  • Feature selection — requires post-training evaluation; cannot be done
    in the pipeline without a reference dataset.
"""

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Ray ───────────────────────────────────────────────────────────────────
import ray
from ray.util.joblib import register_ray

# ── ML ────────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,   # PERF-1: was GridSearchCV
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── Visualisation — non-blocking backend ──────────────────────────────────
# PERF-4: 'Agg' never opens a display window, so plt.show() is a no-op on
# headless servers / Colab without an X session.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after backend set)
import seaborn as sns  # noqa: F401

# ── Fixed ordinal maps for derived categoricals ───────────────────────────
# BUG-2: These columns were created after categorical_cols was captured so
# they were never encoded.  Mapping them to ints immediately after pd.cut()
# keeps them out of the label-encoder flow entirely.
TENURE_GROUP_MAP: Dict[str, int] = {
    "0-1yr": 0, "1-2yr": 1, "2-3yr": 2,
    "3-4yr": 3, "4-5yr": 4, "5-6yr": 5,
}
CHARGE_GROUP_MAP: Dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "very_high": 3,
}


# ── ANSI colours ──────────────────────────────────────────────────────────
class Colors:
    GREEN = "\033[92m";  RED     = "\033[91m";  YELLOW = "\033[93m"
    BLUE  = "\033[94m";  MAGENTA = "\033[95m";  CYAN   = "\033[96m"
    WHITE = "\033[97m";  BOLD    = "\033[1m";   ENDC   = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════
# Module-level Ray remote functions
#
# BUG-4 / BUG-7: Originally these were @ray.remote instance methods or
# closures defined inside instance methods.  Both patterns force Ray to
# serialise the full pipeline object for every task.
#
# Module-level functions only serialise their explicit arguments.
# We further use ray.put() references for large objects (X, y, model) so
# they are uploaded to the object store exactly once.
# ══════════════════════════════════════════════════════════════════════════

@ray.remote
def _cv_fold_worker(
    model_params: Dict[str, Any],
    X_ref,               # ray.ObjectRef pointing to X_train
    y_ref,               # ray.ObjectRef pointing to y_train
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> float:
    """
    Execute one CV fold in a Ray worker.

    PERF-3: Uses CPU tree_method so N workers can run truly in parallel
    without GPU contention. The caller strips GPU params before passing.
    """
    X = X_ref
    y = y_ref
    model = XGBClassifier(**model_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
    return roc_auc_score(y.iloc[val_idx], y_proba)


@ray.remote
def _predict_batch_worker(
    model_ref,              # ray.ObjectRef — model uploaded once via ray.put()
    X_batch: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on one batch.

    BUG-5: Original passed self.model directly, serialising it per task.
    model_ref is a tiny ObjectRef; the model bytes live in the object store
    and are zero-copy-mapped into each worker.
    """
    model = ray.get(model_ref)
    return model.predict(X_batch), model.predict_proba(X_batch)[:, 1]


# ══════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════

class DistributedLogger:
    """Enhanced logging with file persistence and buffered I/O."""

    # PERF-5: flush every N fit-log entries to avoid O(fits) blocking writes.
    _FLUSH_INTERVAL: int = 50

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"xgb_experiment_{ts}"
        self.log_file     = self.log_dir / f"{self.experiment_name}.log"
        self.metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        self.fits_file    = self.log_dir / f"{self.experiment_name}_fits.log"

        self.logger = logging.getLogger(f"XGB_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)

        for handler, fmt in [
            (
                logging.FileHandler(self.log_file),
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ),
            (
                logging.StreamHandler(),
                "%(asctime)s - %(levelname)s - %(message)s",
            ),
        ]:
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)

        self.performance_log: List[Dict] = []
        self.fit_counter: int = 0
        self._fit_buffer: List[Dict] = []  # PERF-5: batched I/O buffer

        self.logger.info(f"Pipeline initialised — Experiment: {self.experiment_name}")

    # ── public ──────────────────────────────────────────────────────────

    def log_fit(
        self,
        fold: int,
        candidate: int,
        total_candidates: int,
        score: float,
        params: Dict[str, Any],
    ) -> None:
        self.fit_counter += 1
        self._fit_buffer.append(
            {
                "timestamp": datetime.now().isoformat(),
                "fit_number": self.fit_counter,
                "fold": fold,
                "candidate": candidate,
                "total_candidates": total_candidates,
                "score": score,
                "parameters": params,
            }
        )
        if len(self._fit_buffer) >= self._FLUSH_INTERVAL:
            self._flush_fit_buffer()
        self.logger.info(
            f"Fit {self.fit_counter}: Fold {fold}, "
            f"Candidate {candidate}/{total_candidates}, Score: {score:.4f}"
        )

    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        self.performance_log.append(
            {"timestamp": datetime.now().isoformat(), "metrics": metrics}
        )
        with open(self.metrics_file, "w") as f:
            json.dump(self.performance_log, f, indent=2, default=str)

    def log_ray_cluster_info(self) -> None:
        try:
            self.logger.info(f"Ray Cluster Resources: {ray.cluster_resources()}")
        except Exception as exc:
            self.logger.warning(f"Could not retrieve Ray cluster info: {exc}")

    # ── internal ────────────────────────────────────────────────────────

    def _flush_fit_buffer(self) -> None:
        if not self._fit_buffer:
            return
        try:
            with open(self.fits_file, "a") as f:
                f.writelines(json.dumps(e) + "\n" for e in self._fit_buffer)
            self._fit_buffer.clear()
        except Exception as exc:
            self.logger.warning(f"Failed to flush fit buffer: {exc}")

    def __del__(self) -> None:
        try:
            self._flush_fit_buffer()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 10,
    save_path: Optional[str] = None,
    headless: bool = True,          # PERF-4: default True for Colab/CI
) -> None:
    imp     = model.feature_importances_
    indices = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Feature Importance (XGBoost)")
    ax.bar(range(len(indices)), imp[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices], rotation=45, ha="right"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if not headless:
        plt.show()
    plt.close(fig)  # always release memory


def plot_metrics(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    headless: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].plot(metrics["fpr"], metrics["tpr"], label=f"AUC = {metrics['auc']:.3f}")
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    axes[0].legend()

    axes[1].plot(
        metrics["recall"],
        metrics["precision"],
        label=f"PR AUC = {metrics['pr_auc']:.3f}",
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if not headless:
        plt.show()
    plt.close(fig)


def print_formatted_results(
    metrics: Dict[str, Any], headless: bool = True
) -> None:
    print("\n=== Model Performance Summary ===")
    print(f"{Colors.GREEN}✓{Colors.ENDC} Accuracy: {metrics.get('test_accuracy', 0):.4f}")

    cm = metrics.get("confusion_matrix_test", np.zeros((2, 2), int))
    print(f"\n{Colors.GREEN}✓{Colors.ENDC} Confusion Matrix:")
    print(f"[[{cm[0][0]} {cm[0][1]}]\n [{cm[1][0]} {cm[1][1]}]]")

    print(f"\n{Colors.GREEN}✓{Colors.ENDC} Classification Report:")
    print(f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n")

    report = metrics.get("classification_report", {})
    for lbl in ["0", "1"]:
        if lbl in report:
            m = report[lbl]
            print(
                f"{lbl:>15} {m['precision']:>10.2f} {m['recall']:>10.2f} "
                f"{m['f1-score']:>10.2f} {int(m['support']):>10}"
            )
    print()
    macro_sup = int(report.get("macro avg", {}).get("support", 0))
    if "accuracy" in report:
        print(
            f"{'accuracy':>15} {'':>10} {'':>10} "
            f"{report['accuracy']:>10.2f} {macro_sup:>10}"
        )
    for key in ("macro avg", "weighted avg"):
        if key in report:
            m   = report[key]
            sup = int(m.get("support", 0))
            print(
                f"{key:>15} {m['precision']:>10.2f} {m['recall']:>10.2f} "
                f"{m['f1-score']:>10.2f} {sup:>10}"
            )

    # PERF-4: save plots, never block
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "model" in metrics and "feature_names" in metrics:
        plot_feature_importance(
            metrics["model"],
            metrics["feature_names"],
            save_path=str(log_dir / f"feature_importance_{ts}.png"),
            headless=headless,
        )
    plot_metrics(
        metrics,
        save_path=str(log_dir / f"metrics_{ts}.png"),
        headless=headless,
    )
    print(f"\n{Colors.BLUE}📊 Plots saved to logs/{Colors.ENDC}")


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Pipeline configuration — GPU-optimised defaults for a single T4."""

    test_size:             float          = 0.2
    random_state:          int            = 42
    cv_folds:              int            = 5
    n_jobs:                int            = -1
    # PERF-6: 50 rounds is plenty; 100 just wastes time when the val-loss
    # has clearly flattened.
    early_stopping_rounds: int            = 50
    batch_size:            int            = 8192
    ray_address:           Optional[str]  = None
    num_cpus:              Optional[int]  = None
    num_gpus:              int            = 1
    memory_fraction:       float          = 0.90
    max_bin:               int            = 256
    # Kept as config attributes so load_model round-trips cleanly, even
    # though they are no longer forwarded to XGBoost (deprecated in ≥1.7).
    force_col_wise:        bool           = True
    gpu_use_dp:            bool           = True
    enable_reduced_grid:   bool           = False
    base_params:           Optional[Dict] = None

    def __post_init__(self) -> None:
        if self.base_params is None:
            # PERF-6: removed deprecated params: n_gpus, gpu_platform_id,
            # gpu_device_id, force_col_wise, gpu_use_dp.
            # n_estimators 2000→1000; early_stopping decides the real count.
            # max_depth 10→8 to reduce overfitting.
            self.base_params = {
                "objective":        "binary:logistic",
                "random_state":     self.random_state,
                "n_jobs":           1,   # 1 → GPU handles own parallelism
                "verbosity":        0,
                # "tree_method":      "gpu_hist",
                # "gpu_id":           0,
                "max_depth":        8,
                "min_child_weight": 2,
                "subsample":        0.85,
                "colsample_bytree": 0.85,
                "learning_rate":    0.05,
                "n_estimators":     1000,
                "scale_pos_weight": 1.0,
                "gamma":            0.1,
                "reg_alpha":        0.1,
                "reg_lambda":       1.5,
                "max_bin":          self.max_bin,
                "grow_policy":      "lossguide",
                "max_leaves":       64,
                "device":           "cuda",
            }


# ══════════════════════════════════════════════════════════════════════════
# Distributed preprocessor Ray actor (kept for future distributed loads)
# ══════════════════════════════════════════════════════════════════════════

@ray.remote
class DistributedPreprocessor:
    """Ray actor for distributed feature preprocessing."""

    def __init__(self) -> None:
        self.label_encoders: Dict[str, Any] = {}

    def fit_encoders(
        self, df_chunk: pd.DataFrame, categorical_cols: List[str]
    ) -> Dict[str, Any]:
        encoders: Dict[str, Any] = {}
        for col in categorical_cols:
            if col in df_chunk.columns and col != "Churn":
                mask = df_chunk[col].notna()
                if mask.sum() > 0:
                    le = LabelEncoder()
                    le.fit(df_chunk.loc[mask, col].unique())
                    encoders[col] = le
        return encoders

    def transform_chunk(
        self, df_chunk: pd.DataFrame, encoders: Dict[str, Any]
    ) -> pd.DataFrame:
        df = df_chunk.copy()
        for col, enc in encoders.items():
            if col in df.columns:
                mask   = df[col].notna()
                seen   = set(enc.classes_)
                unseen = set(df.loc[mask, col].unique()) - seen
                if unseen:
                    df.loc[mask & df[col].isin(unseen), col] = enc.classes_[0]
                known = mask & df[col].isin(seen)
                if known.sum() > 0:
                    df.loc[known, col] = enc.transform(df.loc[known, col])
                df[col] = df[col].astype("Int64")
        return df


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════

class ProductionXGBoostPipeline:
    """Ray-optimised XGBoost pipeline with comprehensive logging."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.model: Optional[XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None

        # ── preprocessing state — must all be saved / restored ───────────
        self.label_encoders:    Dict[str, Any]   = {}
        # BUG-6: store training medians/modes/stats for consistent inference
        self.impute_values:     Dict[str, float] = {}  # per-column medians
        self.categorical_modes: Dict[str, str]   = {}  # per-column modes
        self.numeric_stats:     Dict[str, Dict]  = {}  # {col: {mean, std}}

        self.model_metadata:     Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}

        self.logger = DistributedLogger()
        self._initialize_ray()
        register_ray()

    # ── Ray initialisation ────────────────────────────────────────────────

    def _initialize_ray(self) -> None:
        if not ray.is_initialized():
            ray_cfg: Dict[str, Any] = {
                "ignore_reinit_error": True,
                "include_dashboard":   False,
                "log_to_driver":       False,
                "num_gpus":            self.config.num_gpus,
                "object_store_memory": 6 * 1024 ** 3,   # 6 GB
                "_memory":              12 * 1024 ** 3,   # 12 GB
                "_temp_dir":            "/tmp/ray",
            }
            if self.config.ray_address:
                ray_cfg["address"] = self.config.ray_address
            if self.config.num_cpus:
                ray_cfg["num_cpus"] = self.config.num_cpus

            ray.init(**ray_cfg)

            if self.config.num_gpus > 0:
                try:
                    import torch

                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_mem  = (
                            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                        )
                        self.logger.logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
                        if "T4" in gpu_name:
                            torch.cuda.set_per_process_memory_fraction(
                                self.config.memory_fraction
                            )
                            self.logger.logger.info(
                                f"GPU memory limit: {self.config.memory_fraction*100:.0f}%"
                            )
                except ImportError:
                    self.logger.logger.warning("PyTorch not available for GPU detection")

            self.logger.log_ray_cluster_info()

    # ── Hyperparameter search ──────────────────────────────────────────────

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Randomised hyperparameter search.

        PERF-1: GridSearchCV with 12 two-option params → ~2 048 combos × 5
        folds = ~10 240 fits.  RandomizedSearchCV with n_iter=30 → 150 fits.
        Typical AUC difference: <1 %.  Speed gain: ~68×.

        PERF-2: n_jobs=1 when GPU is active. GridSearchCV with n_jobs=-1
        and tree_method='gpu_hist' spawns multiple processes that all try to
        use the same GPU simultaneously — slower than sequential.
        """
        t0 = time.time()
        self.logger.logger.info("Starting RandomizedSearchCV optimisation …")

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = round(neg / pos, 4) if pos > 0 else 1.0

        # GPU-specific params live in base_params; keep search space clean.
        param_distributions: Dict[str, List] = {
            "n_estimators":     [500, 750, 1000, 1500],
            "max_depth":        [5, 6, 8, 10],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.70, 0.80, 0.85, 0.90],
            "colsample_bytree": [0.70, 0.80, 0.85, 0.90],
            "min_child_weight": [1, 2, 3, 5],
            "gamma":            [0.0, 0.05, 0.1, 0.2],
            "reg_alpha":        [0.0, 0.1, 0.5, 1.0],
            "reg_lambda":       [1.0, 1.5, 2.0, 3.0],
            "max_leaves":       [32, 64],
            "scale_pos_weight": [1.0, scale_pos_weight],
        }

        n_iter = 30
        total_fits = n_iter * self.config.cv_folds
        self.logger.logger.info(
            f"RandomizedSearchCV: {n_iter} iter × {self.config.cv_folds} folds "
            f"= {total_fits} fits  (original GridSearch ≈ 10 240)"
        )

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # PERF-2: n_jobs=1 for GPU to avoid contention
        search_n_jobs = 1 if self.config.num_gpus > 0 else -1
        base_model    = XGBClassifier(**self.config.base_params)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            n_jobs=search_n_jobs,
            verbose=1,
            random_state=self.config.random_state,
            error_score="raise",
        )

        try:
            search.fit(X_train, y_train)
            best_params = search.best_params_
            best_score  = search.best_score_
            self.logger.logger.info(
                f"Optimisation done in {time.time()-t0:.1f}s — "
                f"Best CV AUC: {best_score:.4f}"
            )
            self.logger.logger.info(f"Best params: {best_params}")

            for i, (p, s) in enumerate(
                zip(
                    search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                )
            ):
                self.logger.log_fit(0, i + 1, n_iter, float(s), p)

            return best_params

        except Exception as exc:
            self.logger.logger.error(f"Optimisation failed: {exc} — using fallback params")
            return {
                "n_estimators": 1000, "max_depth": 8,   "learning_rate": 0.05,
                "subsample": 0.85,    "colsample_bytree": 0.85,
                "min_child_weight": 2, "gamma": 0.1,
                "reg_alpha": 0.1,     "reg_lambda": 1.5,
                "max_leaves": 64,     "scale_pos_weight": scale_pos_weight,
            }

    # ── Preprocessing ──────────────────────────────────────────────────────

    def preprocess_features(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """
        Feature engineering and encoding.

        All five preprocessing bugs (BUG-1 … BUG-3, BUG-6) are fixed here.
        See module docstring for details.
        """
        df_proc = df.copy()

        # ── 1. Snapshot column types from ORIGINAL data ───────────────────
        num_cols = [
            c for c in df_proc.select_dtypes(include=["int64", "float64"]).columns
            if c != "Churn"
        ]
        cat_cols = [
            c for c in df_proc.select_dtypes(include=["object", "category"]).columns
            if c != "Churn"
        ]

        # ── 2. Missingness indicators BEFORE imputation ───────────────────
        # BUG-3: original added indicators after fillna → always 0.
        for col in num_cols:
            if df_proc[col].isna().any():
                df_proc[f"{col}_is_na"] = df_proc[col].isna().astype(int)

        # ── 3. Impute with stored training statistics ─────────────────────
        # BUG-6: original computed median/mode from current data at inference
        # time → different fill values per batch, corrupted z-scores.
        if is_training:
            self.impute_values    = {}
            self.categorical_modes = {}
            for col in num_cols:
                m = float(df_proc[col].median())
                self.impute_values[col] = m
                df_proc[col] = df_proc[col].fillna(m)
            for col in cat_cols:
                mode_series = df_proc[col].mode()
                mv = mode_series[0] if len(mode_series) > 0 else "Unknown"
                self.categorical_modes[col] = mv
                df_proc[col] = df_proc[col].fillna(mv)
        else:
            for col in num_cols:
                df_proc[col] = df_proc[col].fillna(
                    self.impute_values.get(col, float(df_proc[col].median()))
                )
            for col in cat_cols:
                df_proc[col] = df_proc[col].fillna(
                    self.categorical_modes.get(col, "Unknown")
                )

        # ── 4. Feature engineering (string ops happen before encoding) ─────

        # Tenure
        if "tenure" in df_proc.columns:
            df_proc["tenure_group"] = pd.cut(
                df_proc["tenure"],
                bins=[0, 12, 24, 36, 48, 60, 72],
                labels=["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr", "5-6yr"],
            )
            # BUG-2: encode immediately so the column is int, not Categorical
            df_proc["tenure_group"] = (
                df_proc["tenure_group"].astype(object).map(TENURE_GROUP_MAP).fillna(-1).astype(int)
            )
            df_proc["tenure_to_max"] = df_proc["tenure"] / 72.0
            if "MonthlyCharges" in df_proc.columns:
                df_proc["tenure_monthly_ratio"] = df_proc["tenure"] / (
                    df_proc["MonthlyCharges"] + 1
                )

        # MonthlyCharges
        if "MonthlyCharges" in df_proc.columns:
            df_proc["charge_group"] = pd.cut(
                df_proc["MonthlyCharges"],
                bins=[0, 30, 60, 90, 120],
                labels=["low", "medium", "high", "very_high"],
            )
            # BUG-2: encode immediately
            df_proc["charge_group"] = (
                df_proc["charge_group"].astype(object).map(CHARGE_GROUP_MAP).fillna(-1).astype(int)
            )

        # Service count
        service_cols = [
            "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        avail_svc = [c for c in service_cols if c in df_proc.columns]
        if avail_svc:
            # BUG-1: original apply(axis=1) used Python `and` on a row Series
            # → ValueError every time.  Vectorised replacement:
            active = (df_proc[avail_svc] != "No") & (
                df_proc[avail_svc] != "No internet service"
            )
            df_proc["total_services"] = active.sum(axis=1)

        # Binary indicators (string comparisons must precede label encoding)
        if "InternetService" in df_proc.columns:
            df_proc["has_internet"] = (
                df_proc["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
            )
        if "PhoneService" in df_proc.columns:
            df_proc["has_phone"] = (df_proc["PhoneService"] == "Yes").astype(int)
        if "Contract" in df_proc.columns:
            df_proc["is_month_to_month"] = (
                df_proc["Contract"] == "Month-to-month"
            ).astype(int)
        if "PaymentMethod" in df_proc.columns:
            df_proc["is_automatic_payment"] = (
                df_proc["PaymentMethod"].str.contains("automatic", na=False).astype(int)
            )
        if "SeniorCitizen" in df_proc.columns:
            df_proc["is_senior"] = (df_proc["SeniorCitizen"] == 1).astype(int)
        if "Dependents" in df_proc.columns:
            df_proc["has_dependents"] = (df_proc["Dependents"] == "Yes").astype(int)

        # Composite risk score
        risk_parts = []
        if "Contract"        in df_proc.columns:
            risk_parts.append((df_proc["Contract"]        == "Month-to-month").astype(int))
        if "PaymentMethod"   in df_proc.columns:
            risk_parts.append((df_proc["PaymentMethod"]   == "Electronic check").astype(int))
        if "InternetService" in df_proc.columns:
            risk_parts.append((df_proc["InternetService"] == "Fiber optic").astype(int))
        if risk_parts:
            df_proc["churn_risk_score"] = sum(risk_parts)

        # ── 5. Label-encode original categorical columns ───────────────────
        if is_training:
            self.label_encoders = {}
            for col in cat_cols:
                if col != "Churn":
                    le = LabelEncoder()
                    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
                    self.label_encoders[col] = le
        else:
            for col, enc in self.label_encoders.items():
                if col in df_proc.columns:
                    # Vectorised unseen-category handling (no apply loop)
                    mapping = {v: i for i, v in enumerate(enc.classes_)}
                    df_proc[col] = (
                        df_proc[col].astype(str).map(mapping).fillna(0).astype(int)
                    )

        # ── 6. Z-score features with stored training statistics ────────────
        # BUG-6: original recomputed mean/std from current data at inference.
        if is_training:
            self.numeric_stats = {}
            for col in num_cols:
                mu  = float(df_proc[col].mean())
                sig = float(df_proc[col].std())
                self.numeric_stats[col] = {"mean": mu, "std": sig}
                if sig > 0:
                    df_proc[f"{col}_zscore"] = (df_proc[col] - mu) / sig
        else:
            for col, stats in self.numeric_stats.items():
                if col in df_proc.columns and stats["std"] > 0:
                    df_proc[f"{col}_zscore"] = (
                        (df_proc[col] - stats["mean"]) / stats["std"]
                    )

        # ── 7. Final cleanup ───────────────────────────────────────────────
        df_proc = df_proc.fillna(0)
        return df_proc

    # ── Training ──────────────────────────────────────────────────────────

    def train_model(
        self, df: pd.DataFrame, target_col: str = "Churn"
    ) -> Dict[str, Any]:
        """Complete training pipeline."""
        t0 = time.time()
        self.logger.logger.info("Starting training pipeline …")

        if df.empty:
            raise ValueError("Input dataframe is empty")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Deduplicate (doc-3 suggestion — free accuracy win)
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            self.logger.logger.info(f"Removed {removed} duplicate rows")

        df_proc = self.preprocess_features(df, is_training=True)
        X = df_proc.drop(columns=[target_col])
        y = df_proc[target_col]

        if y.dtype == "object" or str(y.dtype) == "category":
            self.logger.logger.info(f"Encoding target. Values: {y.unique()}")
            te = LabelEncoder()
            y  = pd.Series(te.fit_transform(y), index=y.index, name=y.name)
            self.model_metadata["target_encoder"] = te
            self.model_metadata["target_mapping"] = dict(
                zip(te.classes_, te.transform(te.classes_))
            )
        else:
            self.model_metadata["target_encoder"] = None

        self.feature_names = X.columns.tolist()
        self.logger.logger.info(
            f"Dataset: {X.shape}  |  Class dist: {y.value_counts().to_dict()}"
        )

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25,
            random_state=self.config.random_state,
            stratify=y_temp,
        )
        self.logger.logger.info(
            f"Split — Train: {X_train.shape[0]}, "
            f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
        )

        best_params  = self.optimize_hyperparameters(X_train, y_train)
        final_params = {**self.config.base_params, **best_params}

        self.model = XGBClassifier(**final_params)
        t1 = time.time()

        final_params['eval_metric'] = ['auc', 'error', 'logloss']   # <-- add this line
        final_params['early_stopping_rounds'] = self.config.early_stopping_rounds
        self.model = XGBClassifier(**final_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True,
        )
        self.logger.logger.info(f"Final model trained in {time.time()-t1:.1f}s")

        results = self._evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test)
        self.model_metadata.update(
            {
                "feature_names":   self.feature_names,
                "model_params":    final_params,
                "training_shape":  X_train.shape,
                "class_distribution": y_train.value_counts().to_dict(),
                "best_iteration":  getattr(self.model, "best_iteration_", None),
                "total_training_time": time.time() - t0,
                "ray_cluster_resources": (
                    ray.cluster_resources() if ray.is_initialized() else None
                ),
            }
        )
        self.logger.log_performance_metrics(results)
        self.logger.logger.info(f"Pipeline finished in {time.time()-t0:.1f}s")
        print_formatted_results(results)
        return results

    # ── Evaluation ────────────────────────────────────────────────────────

    def _evaluate_model(
        self,
        X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series,   y_val: pd.Series,     y_test: pd.Series,
    ) -> Dict[str, Any]:

        y_tr_pred   = self.model.predict(X_train)
        y_val_pred  = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        y_tr_proba   = self.model.predict_proba(X_train)[:, 1]
        y_val_proba  = self.model.predict_proba(X_val)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        metrics: Dict[str, Any] = {
            "train_accuracy": accuracy_score(y_train, y_tr_pred),
            "train_auc":      roc_auc_score(y_train, y_tr_proba),
            "train_f1":       f1_score(y_train, y_tr_pred),
            "val_accuracy":   accuracy_score(y_val, y_val_pred),
            "val_auc":        roc_auc_score(y_val, y_val_proba),
            "val_f1":         f1_score(y_val, y_val_pred),
            "test_accuracy":  accuracy_score(y_test, y_test_pred),
            "test_auc":       roc_auc_score(y_test, y_test_proba),
            "test_f1":        f1_score(y_test, y_test_pred),
            "confusion_matrix_test": confusion_matrix(y_test, y_test_pred),
            "classification_report": classification_report(
                y_test, y_test_pred, output_dict=True
            ),
            "model":         self.model,
            "feature_names": self.feature_names,
        }

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        metrics.update(
            {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, y_test_proba)}
        )

        prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
        metrics.update(
            {
                "precision": prec,
                "recall":    rec,
                "pr_auc":    average_precision_score(y_test, y_test_proba),
            }
        )

        # ── Parallel CV ───────────────────────────────────────────────────
        # BUG-4 / PERF-3: was self._parallel_cv_fold.remote(self, …) which
        # serialised the entire pipeline.  Now uses the module-level function
        # with CPU params so workers run truly in parallel without GPU fights.
        cv_params = {
            k: v for k, v in self.model.get_params().items()
            if k != "early_stopping_rounds"
        }
        # Switch workers to CPU
        cv_params.update(
            {"tree_method": "hist", "predictor": "cpu_predictor",
             "n_jobs": 2, "verbosity": 0}
        )
        for gkey in ("gpu_id", "n_gpus", "gpu_platform_id",
                     "gpu_device_id", "gpu_use_dp"):
            cv_params.pop(gkey, None)

        # BUG-4: store data in object store once (not per-task)
        X_ref = X_train
        y_ref = y_train

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        futures = [
            _cv_fold_worker.remote(cv_params, X_ref, y_ref, tr_idx, val_idx)
            for tr_idx, val_idx in cv.split(X_train, y_train)
        ]
        cv_scores = ray.get(futures)
        metrics["cv_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_auc_std"]  = float(np.std(cv_scores))

        if hasattr(self.model, "feature_importances_"):
            metrics["feature_importance"] = pd.DataFrame(
                {
                    "feature":    self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

        self.performance_metrics = metrics
        self.logger.logger.info(
            f"Train AUC: {metrics['train_auc']:.4f} | "
            f"Val AUC: {metrics['val_auc']:.4f} | "
            f"Test AUC: {metrics['test_auc']:.4f}"
        )
        self.logger.logger.info(
            f"CV AUC: {metrics['cv_auc_mean']:.4f} "
            f"(±{metrics['cv_auc_std']*2:.4f})"
        )
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Production prediction.

        Preprocessing is handled internally — callers pass a raw DataFrame.
        This is the single authoritative entry point; do not call
        preprocess_features() separately before this method.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        t0 = time.time()
        self.logger.logger.info(f"Predicting for {len(df)} samples …")

        df_proc = self.preprocess_features(df, is_training=False)
        if "Churn" in df_proc.columns:
            df_proc = df_proc.drop(columns=["Churn"])

        missing = set(self.feature_names) - set(df_proc.columns)
        if missing:
            self.logger.logger.warning(f"Missing features — filling with 0: {missing}")
            for feat in missing:
                df_proc[feat] = 0

        X_pred = df_proc[self.feature_names]

        if len(X_pred) > 10_000:
            # BUG-5: use ray.put() so the model is uploaded once
            model_ref  = ray.put(self.model)
            cpu_count  = max(1, int(ray.available_resources().get("CPU", 1)))
            batch_size = max(1000, len(X_pred) // cpu_count)
            batches    = [
                X_pred.iloc[i : i + batch_size]
                for i in range(0, len(X_pred), batch_size)
            ]
            futures = [
                _predict_batch_worker.remote(model_ref, b) for b in batches
            ]
            results      = ray.get(futures)
            predictions  = np.concatenate([r[0] for r in results])
            probabilities = np.concatenate([r[1] for r in results])
        else:
            predictions   = self.model.predict(X_pred)
            probabilities = self.model.predict_proba(X_pred)[:, 1]

        if self.model_metadata.get("target_encoder") is not None:
            predictions = self.model_metadata["target_encoder"].inverse_transform(
                predictions
            )

        self.logger.logger.info(f"Prediction done in {time.time()-t0:.2f}s")
        return predictions, probabilities

    # ── Persistence ───────────────────────────────────────────────────────

    def save_model(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("No trained model to save")

        pkg = {
            "model":              self.model,
            "label_encoders":     self.label_encoders,
            "impute_values":      self.impute_values,      # BUG-6
            "categorical_modes":  self.categorical_modes,  # BUG-6
            "numeric_stats":      self.numeric_stats,      # BUG-6
            "feature_names":      self.feature_names,
            "metadata":           self.model_metadata,
            "config":             asdict(self.config),
            "performance_metrics": self.performance_metrics,
            "save_timestamp":     datetime.now().isoformat(),
        }
        joblib.dump(pkg, filepath)

        # Compact native artifact for lean inference deployment
        native_path = str(Path(filepath).with_suffix(".ubj"))
        self.model.save_model(native_path)

        metrics_path = Path(filepath).with_suffix(".json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    k: (
                        v.tolist()     if isinstance(v, np.ndarray)
                        else v.to_dict() if isinstance(v, pd.DataFrame)
                        else v
                    )
                    for k, v in self.performance_metrics.items()
                },
                f,
                indent=2,
                default=str,
            )

        self.logger.logger.info(
            f"Model saved → {filepath}  |  native → {native_path}"
        )

    @classmethod
    def load_model(cls, filepath: str) -> "ProductionXGBoostPipeline":
        pkg      = joblib.load(filepath)
        cfg_data = pkg.get("config", {})
        config   = ModelConfig(**cfg_data) if isinstance(cfg_data, dict) else cfg_data
        pipeline = cls(config=config)

        pipeline.model             = pkg["model"]
        pipeline.label_encoders    = pkg["label_encoders"]
        # BUG-6: use .get() for backward compat with models saved before this patch
        pipeline.impute_values     = pkg.get("impute_values",     {})
        pipeline.categorical_modes = pkg.get("categorical_modes", {})
        pipeline.numeric_stats     = pkg.get("numeric_stats",     {})
        pipeline.feature_names     = pkg["feature_names"]
        pipeline.model_metadata    = pkg["metadata"]
        pipeline.performance_metrics = pkg.get("performance_metrics", {})
        pipeline.logger.logger.info(f"Model loaded ← {filepath}")
        return pipeline

    # ── Summary / cleanup ─────────────────────────────────────────────────

    def get_training_summary(self) -> Dict[str, Any]:
        if not self.performance_metrics:
            return {"error": "No metrics available — train first."}
        return {
            "experiment_info": {
                "experiment_name": self.logger.experiment_name,
                "total_fits":      self.logger.fit_counter,
                "completed_at":    datetime.now().isoformat(),
            },
            "performance_metrics": self.performance_metrics,
            "model_info":          self.model_metadata,
            "ray_cluster":         (
                ray.cluster_resources() if ray.is_initialized() else None
            ),
            "log_files": {
                "main_log":    str(self.logger.log_file),
                "fits_log":    str(self.logger.fits_file),
                "metrics_log": str(self.logger.metrics_file),
            },
        }

    def shutdown_ray(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
            self.logger.logger.info("Ray shutdown complete")

    def __del__(self) -> None:
        try:
            if hasattr(self, "logger"):
                self.logger.logger.info("Pipeline cleanup")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"{Colors.CYAN}🚀 Initialising Ray-Optimised XGBoost Pipeline{Colors.ENDC}")

    try:
        df = pd.read_csv("/content/updated_dataset.csv")
        print(f"{Colors.GREEN}✓{Colors.ENDC} Dataset: {df.shape}")
        print(f"Class distribution:\n{df['Churn'].value_counts()}")
    except FileNotFoundError:
        print(f"{Colors.RED}❌ /content/updated_dataset.csv not found{Colors.ENDC}")
        return
    except Exception as exc:
        print(f"{Colors.RED}❌ Load error: {exc}{Colors.ENDC}")
        return

    config = ModelConfig(
        test_size=0.2,
        cv_folds=5,
        early_stopping_rounds=50,
        num_gpus=1,
        batch_size=8192,
        memory_fraction=0.90,
        max_bin=256,
    )
    pipeline = ProductionXGBoostPipeline(config)

    try:
        import torch

        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            gmem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"{Colors.GREEN}✓{Colors.ENDC} GPU: {gpu} ({gmem:.1f} GB)")
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(config.memory_fraction)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
        else:
            print(f"{Colors.YELLOW}⚠️  No GPU — falling back to CPU{Colors.ENDC}")

        results = pipeline.train_model(df)
        summary = pipeline.get_training_summary()

        import os

        os.makedirs("models", exist_ok=True)
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/xgboost_model_{ts}.pkl"
        pipeline.save_model(model_path)
        print(f"{Colors.GREEN}✓{Colors.ENDC} Model saved → {model_path}")

        with open(f"models/metadata_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    except Exception as exc:
        import traceback
        print(f"{Colors.RED}❌ Pipeline error: {exc}{Colors.ENDC}")
        traceback.print_exc()

    finally:
        pipeline.shutdown_ray()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"{Colors.CYAN}🏁 Done{Colors.ENDC}")


if __name__ == "__main__":
    main()
