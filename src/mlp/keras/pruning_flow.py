"""
Repeated train / prune / fine-tune benchmark with stratified splits and seed averaging.

Requires a *builder* that returns a fresh compiled ``Sequential`` each call so every
seed starts from uninitialized weights.
"""

from __future__ import annotations

import logging
import random
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tqdm.auto import tqdm

from importance_pruner import (
    HardMaskCallback,
    HiddenWeightTracker,
    compute_all_importance,
    compute_edge_importance,
    prune_edges_unstructured,
)
from utils.logging_utils import log_or_print, setup_logger

_NUMERIC_METRIC_KEYS = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "sparsity",
)

_COUNT_KEYS = ("tp", "tn", "fp", "fn")

logger = setup_logger()


def _set_global_seeds(seed: int) -> None:
    """
    Set the global random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def capture_metrics(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    label: Literal["full", "pruned"],
) -> Dict[str, Any]:
    """
    Evaluate + confusion matrix + weight sparsity.

    Parameters
    ----------
    model : Sequential
        The model to evaluate.
    X : np.ndarray
        The input test data.
    y : np.ndarray
        The input test labels.
    label : str
        The label for the model.

    Returns
    -------
    Dict[str, Any]
        A dictionary of metrics.

    """
    loss, acc, prec, rec, roc, pr = model.evaluate(X, y, verbose=0)
    probs = model.predict(X, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    y_flat = np.asarray(y).ravel()
    tn, fp, fn, tp = confusion_matrix(y_flat, preds, labels=[0, 1]).ravel()

    total, zeros = 0, 0
    for layer in model.layers:
        ws = layer.get_weights()
        if ws:
            w = ws[0]
            total += w.size
            zeros += int(np.sum(w == 0))
    sparsity = zeros / total if total > 0 else 0.0

    return {
        "model": label,
        "loss": float(round(loss, 4)),
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "roc_auc": float(round(roc, 4)),
        "pr_auc": float(round(pr, 4)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sparsity": float(round(sparsity * 100, 1)),
    }


def run_pruning_seeded_benchmark(
    build_model: Callable[[], Sequential],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_seeds: int = 10,
    test_size: float = 0.2,
    pre_prune_epochs: int = 65,
    post_prune_epochs: int = 35,
    prune_ratio: float = 0.2,
    hard_masking: bool = False,
    fit_verbose: int = 0,
    fitness_kw: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each seed: runs a training of a full model, a pruned one (train + prune + fine-tune),
    and collects test metrics. Aggregates across seeds, to reduce the effect of random noise of evaluation.

    # Parameters:
    -------------
    build_model : Callable[[], Sequential]
        Zero-argument factory returning a **compiled** ``Sequential`` (same architecture each time).
    X : np.ndarray
        Feature matrix and binary labels (1-D or column vector).
    y : np.ndarray
        Binary labels (1-D or column vector).
    n_seeds : int, optional
        Number of  different random seeds to use (and independent initializations).
        *Default is 1*
    test_size : float, optional
        Held-out fraction for evaluation (stratified).
        *Default is 0.2*
    pre_prune_epochs : int, optional
        Phase 1 epoch count (full model trains for the sum of pre_ and post_prune_epochs).
        *Default is 65*
    post_prune_epochs : int, optional
        Phase 2 epoch count (full model trains for the sum of pre_ and post_prune_epochs).
        *Default is 35*
    prune_ratio : float, optional
        Fraction of lowest-importance edges to zero (see ``importance_pruner.prune_edges_unstructured``).
        *Default is 0.2*
    hard_masking : bool, optional
        If True, fine-tuning uses ``HardMaskCallback``, that results in pruned edges not reactivating.
        *Default is False*
    fit_verbose : int, optional
        Keras ``verbose`` for ``fit`` calls.
        *Default is 0*
    fitness_kw : dict, optional
        Optional extra kwargs forwarded to ``compute_all_importance`` / graph metrics.
        *Default is None*

    # Returns:
    ----------
    summary_df : pd.DataFrame
        One row per metric (scores and confusion counts averaged over all seeds).
    per_seed_df : pd.DataFrame
        One row per seed with ``full_*`` and ``pruned_*`` columns for every metric.
    """
    y = np.asarray(y)
    y_1d = y.ravel()

    total_epochs = (
        pre_prune_epochs + post_prune_epochs
    )  # epochs for training the models
    fitness_kw = dict(fitness_kw or {})

    full_rows: List[Dict[str, Any]] = []
    pruned_rows: List[Dict[str, Any]] = []

    updates_per_seed = 5
    total_updates = n_seeds * updates_per_seed  # 5 updates per seed

    pbar = tqdm(total=total_updates, desc="Seeds")

    for seed in range(1, n_seeds + 1):
        pbar.set_description(f"Running seed {seed}/{n_seeds}")
        _set_global_seeds(seed)
        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y_1d,
            random_state=seed,
        )
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        _set_global_seeds(seed)
        full_model = build_model()
        initial_weights = full_model.get_weights()
        pbar.update(1)

        pbar.set_description(f"Training full model (seed {seed}/{n_seeds})...")
        full_model.fit(
            train_X,
            train_y,
            epochs=total_epochs,
            validation_data=(test_X, test_y),
            verbose=fit_verbose,
        )
        full_rows.append(capture_metrics(full_model, test_X, test_y, "full"))
        pbar.update(1)

        pbar.set_description(f"Training pruned model - full (seed {seed}/{n_seeds})...")
        _set_global_seeds(seed)
        phase1_model = build_model()
        phase1_model.set_weights(initial_weights)
        tracker = HiddenWeightTracker(record_epochs=pre_prune_epochs)
        phase1_model.fit(
            train_X,
            train_y,
            epochs=pre_prune_epochs,
            validation_data=(test_X, test_y),
            callbacks=[tracker],
            verbose=fit_verbose,
        )
        pbar.update(1)

        pbar.set_description(f"Computing importances (seed {seed}/{n_seeds})...")
        fitness_dict = compute_all_importance(tracker.snapshots, **fitness_kw)
        importance = compute_edge_importance(tracker.snapshots, fitness_dict)
        pruned_clone, _, _, _, kernel_masks = prune_edges_unstructured(
            phase1_model,
            importance,
            prune_ratio=prune_ratio,
            hard_masking=hard_masking,
        )
        pbar.update(1)

        pbar.set_description(
            f"Training pruned model - pruned (seed {seed}/{n_seeds})..."
        )
        fine_tune_model = build_model()
        fine_tune_model.set_weights(pruned_clone.get_weights())
        ft_callbacks = (
            [HardMaskCallback(fine_tune_model, kernel_masks)]
            if hard_masking and kernel_masks is not None
            else None
        )
        fine_tune_model.fit(
            train_X,
            train_y,
            epochs=post_prune_epochs,
            validation_data=(test_X, test_y),
            callbacks=ft_callbacks,
            verbose=fit_verbose,
        )
        pruned_rows.append(capture_metrics(fine_tune_model, test_X, test_y, "pruned"))
        pbar.update(1)

        del full_model, phase1_model, pruned_clone, fine_tune_model
        tf.keras.backend.clear_session()

    per_seed_records: List[Dict[str, Any]] = []
    for seed, (fr, pr_) in enumerate(
        tqdm(
            zip(full_rows, pruned_rows),
            total=len(full_rows),
            desc="Saving Per Seed DataFrame",
        )
    ):
        row: Dict[str, Any] = {"seed": seed}
        for k in _NUMERIC_METRIC_KEYS:
            row[f"full_{k}"] = fr[k]
            row[f"pruned_{k}"] = pr_[k]
        for k in _COUNT_KEYS:
            row[f"full_{k}"] = fr[k]
            row[f"pruned_{k}"] = pr_[k]
        per_seed_records.append(row)

    per_seed_df = pd.DataFrame(per_seed_records)

    summary_records: List[Dict[str, Any]] = []
    for k in tqdm(
        chain(_NUMERIC_METRIC_KEYS, _COUNT_KEYS),
        total=len(_NUMERIC_METRIC_KEYS) + len(_COUNT_KEYS),
        desc="Creating Summary DataFrame",
    ):
        f = per_seed_df[f"full_{k}"].astype(float)
        p = per_seed_df[f"pruned_{k}"].astype(float)
        summary_records.append(
            {
                "metric": k,
                "full_mean": f.mean(),
                "full_std": f.std(ddof=0),
                "pruned_mean": p.mean(),
                "pruned_std": p.std(ddof=0),
            }
        )
    summary_df = pd.DataFrame(summary_records)
    return summary_df, per_seed_df


__all__ = [
    "capture_metrics",
    "run_pruning_seeded_benchmark",
]
