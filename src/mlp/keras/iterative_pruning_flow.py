"""
Repeated train / iteratively-prune / fine-tune benchmark with stratified splits
and seed averaging.

Same seeded benchmark as ``pruning_flow``, but the one-shot
"train -> score -> cut once -> fine-tune" pruning step is replaced by
``iterative_pruner.iterative_prune``: pre-train, then repeat (score the
current subnetwork -> cut a scheduled fraction of the *surviving* structure
-> fine-tune) ``n_rounds`` times until ``target_sparsity`` is reached.

Requires a *builder* that returns a fresh compiled ``Sequential`` each call so every
seed starts from uninitialized weights.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from itertools import chain
from typing import Any, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tqdm.auto import tqdm

try:
    from iterative_pruner import iterative_prune
    from utils.logging_utils import log_or_print, setup_logger
except ImportError:
    from src.mlp.keras.iterative_pruner import iterative_prune
    from src.utils.logging_utils import log_or_print, setup_logger

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
) -> dict[str, Any]:
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
    n_rounds: int = 5,
    target_sparsity: float = 0.2,
    schedule: Literal["linear", "cubic"] = "cubic",
    granularity: Literal["edges", "neurons"] = "edges",
    hard_masking: bool = True,
    rollback_threshold: float | None = None,
    rollback_metric: str = "loss",
    rollback_mode: Literal["min", "max", "auto"] = "auto",
    reconstruct_model: bool = True,
    fit_verbose: int = 0,
    fitness_kw: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each seed: runs a training of a full model, an iteratively pruned one
    (pre-train, then ``n_rounds`` rounds of score / cut a fraction of the
    survivors / fine-tune via ``iterative_pruner.iterative_prune``), and collects
    test metrics. Aggregates across seeds, to reduce the effect of random noise
    of evaluation.

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
        *Default is 10*
    test_size : float, optional
        Held-out fraction for evaluation (stratified).
        *Default is 0.2*
    pre_prune_epochs : int, optional
        Phase 1 epoch count: the pre-training window of the iterative run
        (the full model trains for the sum of pre_ and post_prune_epochs).
        *Default is 65*
    post_prune_epochs : int, optional
        Total fine-tuning budget of the iterative run, split across the
        ``n_rounds`` rounds: each round fine-tunes for
        ``post_prune_epochs // n_rounds`` epochs (at least 2, so the temporal
        importance terms do not degenerate).
        *Default is 35*
    n_rounds : int, optional
        Maximum number of iterative cut rounds ``R`` (see
        ``iterative_pruner.sparsity_schedule``).
        *Default is 5*
    target_sparsity : float, optional
        Cumulative sparsity the iterative run reaches after ``n_rounds`` rounds
        (replaces the one-shot ``prune_ratio``), strictly between 0 and 1.
        *Default is 0.2*
    schedule : str, optional
        ``"linear"`` or ``"cubic"`` cumulative-sparsity schedule.
        *Default is "cubic"*
    granularity : str, optional
        ``"edges"`` (unstructured) or ``"neurons"`` (structured) pruning.
        *Default is "edges"*
    hard_masking : bool, optional
        If True, fine-tuning enforces the cumulative masks, so pruned structure
        never regrows within or across rounds.
        *Default is True*
    rollback_threshold : float or None, optional
        Stop a seed's iterative run early and restore its best state when
        ``rollback_metric`` degrades by more than this much relative to its best
        value (``None`` disables rollback).
        *Default is None*
    rollback_metric : str, optional
        Metric name from ``model.evaluate`` used for rollback (e.g. ``"loss"``).
        *Default is "loss"*
    rollback_mode : str, optional
        ``"min"`` (lower is better), ``"max"``, or ``"auto"`` direction of
        ``rollback_metric``.
        *Default is "auto"*
    reconstruct_model : bool, optional
        Physically remove dead structure from the final pruned model (see
        ``importance_pruner.reconstruct``).
        *Default is True*
    fit_verbose : int, optional
        Keras ``verbose`` for ``fit`` calls.
        *Default is 0*
    fitness_kw : dict, optional
        Optional extra kwargs forwarded to ``compute_all_importance`` / graph metrics.
        *Default is None*
    fit_kwargs : dict, optional
        Extra kwargs forwarded to every ``fit`` call (e.g. ``batch_size``); must
        not contain ``epochs``, ``verbose``, ``callbacks`` or ``validation_data``.
        *Default is None*

    # Returns:
    ----------
    summary_df : pd.DataFrame
        One row per metric (scores and confusion counts averaged over all seeds).
    per_seed_df : pd.DataFrame
        One row per seed with ``full_*`` and ``pruned_*`` columns for every
        metric, plus the ``pruned_rounds_executed`` and ``pruned_rolled_back``
        diagnostics.  The pruned ``sparsity`` is the cumulative sparsity
        reported by ``iterative_prune`` (measured before any reconstruction),
        so it stays meaningful even when the final model was reconstructed.
    """
    if n_rounds < 1:
        log_or_print(f"n_rounds must be >= 1, got {n_rounds}.", level="error")
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if not 0.0 < target_sparsity < 1.0:
        log_or_print(
            f"target_sparsity must be in (0, 1), got {target_sparsity}.",
            level="error",
        )
        raise ValueError(f"target_sparsity must be in (0, 1), got {target_sparsity}.")
    if schedule not in ("linear", "cubic"):
        log_or_print(
            f"schedule must be 'linear' or 'cubic', got '{schedule}'.",
            level="error",
        )
        raise ValueError(f"schedule must be 'linear' or 'cubic', got '{schedule}'.")
    if granularity not in ("edges", "neurons"):
        log_or_print(
            f"granularity must be 'edges' or 'neurons', got '{granularity}'.",
            level="error",
        )
        raise ValueError(
            f"granularity must be 'edges' or 'neurons', got '{granularity}'."
        )
    if pre_prune_epochs < 1:
        log_or_print(
            f"pre_prune_epochs must be >= 1, got {pre_prune_epochs}.", level="error"
        )
        raise ValueError(f"pre_prune_epochs must be >= 1, got {pre_prune_epochs}.")
    fit_kwargs = dict(fit_kwargs or {})
    reserved_fit_kwargs = fit_kwargs.keys() & {
        "epochs",
        "verbose",
        "callbacks",
        "validation_data",
    }
    if reserved_fit_kwargs:
        msg = (
            f"fit_kwargs must not contain {sorted(reserved_fit_kwargs)}; "
            "they are set by the benchmark and iterative_prune."
        )
        log_or_print(msg, level="error")
        raise ValueError(msg)

    y = np.asarray(y)
    y_1d = y.ravel()

    total_epochs = pre_prune_epochs + post_prune_epochs
    fine_tune_epochs = max(2, post_prune_epochs // n_rounds)
    if fine_tune_epochs * n_rounds != post_prune_epochs:
        log_or_print(
            f"post_prune_epochs ({post_prune_epochs}) does not split evenly into "
            f"{n_rounds} rounds; each round fine-tunes for {fine_tune_epochs} "
            f"epochs (total {n_rounds * fine_tune_epochs}).",
            level="warning",
        )
    fitness_kw = dict(fitness_kw or {})

    full_rows: list[dict[str, Any]] = []
    pruned_rows: list[dict[str, Any]] = []
    rounds_executed: list[int] = []
    rolled_back: list[bool] = []

    updates_per_seed = 3
    total_updates = n_seeds * updates_per_seed

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

        pbar.set_description(f"Iteratively pruning model (seed {seed}/{n_seeds})...")
        _set_global_seeds(seed)
        phase1_model = build_model()
        phase1_model.set_weights(initial_weights)
        result = iterative_prune(
            phase1_model,
            train_X,
            train_y,
            target_sparsity=target_sparsity,
            n_rounds=n_rounds,
            schedule=schedule,
            granularity=granularity,
            pretrain_epochs=pre_prune_epochs,
            fine_tune_epochs=fine_tune_epochs,
            validation_data=(test_X, test_y),
            fitness_kw=fitness_kw,
            hard_masking=hard_masking,
            rollback_threshold=rollback_threshold,
            rollback_metric=rollback_metric,
            rollback_mode=rollback_mode,
            reconstruct_model=reconstruct_model,
            fit_verbose=fit_verbose,
            fit_kwargs=fit_kwargs,
        )

        pruned_row = capture_metrics(result.model, test_X, test_y, "pruned")
        sparsity_fraction = (
            result.final_neuron_sparsity
            if granularity == "neurons" and result.final_neuron_sparsity is not None
            else result.final_edge_sparsity
        )
        pruned_row["sparsity"] = float(round(sparsity_fraction * 100.0, 1))
        pruned_rows.append(pruned_row)
        rounds_executed.append(result.n_rounds_executed)
        rolled_back.append(result.rolled_back)
        pbar.update(1)

        del full_model, phase1_model, result
        tf.keras.backend.clear_session()

    per_seed_records: list[dict[str, Any]] = []
    for seed, fr, pr_, n_exec, rb in tqdm(
        zip(
            range(1, n_seeds + 1),
            full_rows,
            pruned_rows,
            rounds_executed,
            rolled_back,
        ),
        total=n_seeds,
        desc="Saving Per Seed DataFrame",
    ):
        row: dict[str, Any] = {"seed": seed}
        for k in _NUMERIC_METRIC_KEYS:
            row[f"full_{k}"] = fr[k]
            row[f"pruned_{k}"] = pr_[k]
        for k in _COUNT_KEYS:
            row[f"full_{k}"] = fr[k]
            row[f"pruned_{k}"] = pr_[k]
        row["pruned_rounds_executed"] = n_exec
        row["pruned_rolled_back"] = rb
        per_seed_records.append(row)

    per_seed_df = pd.DataFrame(per_seed_records)

    summary_records: list[dict[str, Any]] = []
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
