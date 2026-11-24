"""
Iterative (multi-round) relationship-aware pruning.

Turns the one-shot "train -> score -> cut once -> fine-tune" pipeline of
``importance_pruner`` into a gradual loop in the spirit of iterative
magnitude pruning (Frankle & Carbin, 2019) and gradual pruning schedules
(Zhu & Gupta, 2019; Gale et al., 2019):

    pre-train
      -> [ score on the current subnetwork
           -> cut a small fraction of the *surviving* structure
           -> fine-tune under hard masks (fresh snapshot window) ] x R
      -> reconstruct (optional)

Cumulative sparsity follows a schedule (linear or cubic).  The fraction
removed in round *r* is measured against the *surviving* structure only,
so cuts are monotone: dead edges / neurons never soak up the selection
percentile and, while hard masking is on, never come back.

Because ``HiddenWeightTracker`` snapshots are re-collected during every
round's fine-tuning, the temporal metrics (strength, plasticity,
centrality, stability, concentration) are re-derived on the *compensating*
subnetwork rather than on the original one.

Main entry points:
------------------
+ ``sparsity_schedule``     – cumulative sparsity targets per round.
+ ``survivor_fraction``     – fraction of survivors to cut in one round.
+ ``select_weak_live``      – weak-neuron selection among live neurons only.
+ ``cut_surviving_edges``   – survivor-aware, monotone edge cut.
+ ``neuron_masks``          – kernel/bias masks that keep dead neurons dead.
+ ``MaskCallback``          – callback enforcing kernel/bias masks while
  fine-tuning (generalizes ``importance_pruner.HardMaskCallback``).
+ ``iterative_prune``       – the iterative driver (edge or neuron level).
+ ``IterativePruneResult``  – what ``iterative_prune`` returns.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model

try:
    from importance_pruner import (
        HiddenWeightTracker,
        compute_all_importance,
        compute_edge_importance,
        prune_neurons,
        reconstruct,
        restorable_compile_config,
    )
    from utils.logging_utils import log_or_print
except ImportError:
    from src.mlp.keras.importance_pruner import (
        HiddenWeightTracker,
        compute_all_importance,
        compute_edge_importance,
        prune_neurons,
        reconstruct,
        restorable_compile_config,
    )
    from src.utils.logging_utils import log_or_print

__all__ = [
    "MaskCallback",
    "IterativePruneResult",
    "cut_surviving_edges",
    "iterative_prune",
    "neuron_masks",
    "select_weak_live",
    "sparsity_schedule",
    "survivor_fraction",
]


def sparsity_schedule(
    target_sparsity: float,
    n_rounds: int,
    schedule: Literal["linear", "cubic"] = "cubic",
) -> np.ndarray:
    """
    Cumulative sparsity targets ``s_1 .. s_R`` for an iterative run.

    Parameters:
    -----------
    target_sparsity : float
        Final cumulative sparsity to reach after ``n_rounds`` rounds
        (strictly between 0 and 1).
    n_rounds : int
        Number of rounds ``R`` (>= 1).
    schedule : str, optional
        ``"linear"`` – equal steps, ``s_r = S * r / R``.
        ``"cubic"`` (default) – ``s_r = S * (1 - (1 - r/R)^3)``, a
        front-loaded schedule that cuts aggressively early and eases into
        the target (Zhu & Gupta, 2019; Gale et al., 2019).

    Returns:
    --------
    np.ndarray
        1-D array of length ``n_rounds`` with monotonically increasing
        cumulative targets in ``(0, 1)``.
    """

    if not 0.0 < target_sparsity < 1.0:
        log_or_print(
            f"target_sparsity must be in (0, 1), got {target_sparsity}.",
            level="error",
        )
        raise ValueError(f"target_sparsity must be in (0, 1), got {target_sparsity}.")
    if n_rounds < 1:
        log_or_print(f"n_rounds must be >= 1, got {n_rounds}.", level="error")
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if schedule not in ("linear", "cubic"):
        log_or_print(
            f"schedule must be 'linear' or 'cubic', got '{schedule}'.",
            level="error",
        )
        raise ValueError(f"schedule must be 'linear' or 'cubic', got '{schedule}'.")

    t = np.arange(1, n_rounds + 1, dtype=float) / n_rounds
    if schedule == "linear":
        return target_sparsity * t
    return target_sparsity * (1.0 - (1.0 - t) ** 3)


def survivor_fraction(prev_sparsity: float, round_target: float) -> float:
    """
    Fraction of *surviving* structure to remove in one round.

    Solves ``prev + f * (1 - prev) = round_target`` for ``f``, i.e.
    ``f = (round_target - prev) / (1 - prev)``.

    Parameters:
    -----------
    prev_sparsity : float
        Cumulative sparsity before the round (fraction in [0, 1]).
    round_target : float
        Cumulative sparsity the round should reach (fraction in [0, 1]).

    Returns:
    --------
    float
        Fraction in ``[0, 1]``; ``0.0`` when the target has already been
        reached (or nothing survives), capped at ``1.0``.
    """

    if round_target <= prev_sparsity:
        return 0.0
    remaining = 1.0 - prev_sparsity
    if remaining <= 0.0:
        return 0.0
    return float(min((round_target - prev_sparsity) / remaining, 1.0))


def select_weak_live(
    importance: np.ndarray,
    dead_idx: Sequence[int] | np.ndarray | None = None,
    prune_ratio: float = 0.2,
) -> np.ndarray:
    """
    Indices of the weakest ``prune_ratio`` fraction of *live* neurons.

    Unlike ``importance_pruner.select_weak``, already-dead neurons are
    excluded from the ranking, so a round's removal budget is never spent
    re-selecting structure that is already gone.

    Parameters:
    -----------
    importance : np.ndarray
        Importance values of all neurons in the layer.
    dead_idx : Sequence[int] or np.ndarray, optional
        Indices of already-dead neurons (excluded from selection).
    prune_ratio : float, optional
        Fraction of the *live* neurons to select, by default 0.2.

    Returns:
    --------
    np.ndarray
        Sorted array of original indices (possibly empty).
    """

    importance = np.asarray(importance, dtype=float)
    n = importance.shape[0]
    if n == 0:
        return np.asarray([], dtype=int)
    if dead_idx is None:
        dead = np.asarray([], dtype=int)
    else:
        dead = np.unique(np.asarray(list(dead_idx), dtype=int))
    live = np.setdiff1d(np.arange(n), dead)
    k = min(int(live.size * prune_ratio), live.size)
    if k <= 0:
        return np.asarray([], dtype=int)
    weak = live[np.argsort(importance[live])[:k]]
    return np.sort(weak)


def cut_surviving_edges(
    edge_importance: dict[int, np.ndarray],
    alive: dict[int, np.ndarray],
    fraction: float,
) -> dict[int, np.ndarray]:
    """
    Survivor-aware, monotone keep-masks for one round of edge pruning.

    The removal threshold is the ``fraction`` percentile of importance
    scores over *surviving* edges only, and the returned masks are
    ``alive & (score >= threshold)`` — edges killed in earlier rounds stay
    dead and never soak up the percentile.

    Parameters:
    -----------
    edge_importance : Dict[int, np.ndarray]
        Edge importance per hidden layer (e.g. from
        ``compute_edge_importance``), keyed ``0 .. n_hidden - 1``.
    alive : Dict[int, np.ndarray]
        Boolean keep-masks of currently-live edges per hidden layer,
        keyed ``0 .. n_hidden - 1``.
    fraction : float
        Fraction of the *surviving* edges to remove, in ``(0, 1]``.

    Returns:
    --------
    Dict[int, np.ndarray]
        New boolean keep-masks per hidden layer.
    """

    if not 0.0 < fraction <= 1.0:
        log_or_print(f"fraction must be in (0, 1], got {fraction}.", level="error")
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    n_layers = len(alive)
    if n_layers == 0:
        log_or_print("alive contains no layers.", level="error")
        raise ValueError("alive contains no layers.")
    live_scores = np.concatenate(
        [
            np.asarray(edge_importance[i])[np.asarray(alive[i], dtype=bool)]
            for i in range(n_layers)
        ]
    )
    if live_scores.size == 0:
        log_or_print("No surviving edges left to prune.", level="error")
        raise ValueError("No surviving edges left to prune.")
    threshold = float(np.percentile(live_scores, fraction * 100.0))
    return {
        i: (
            np.asarray(alive[i], dtype=bool)
            & (np.asarray(edge_importance[i]) >= threshold)
        )
        for i in range(n_layers)
    }


def neuron_masks(
    model: Sequential,
    dead_neurons: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Kernel and bias masks that keep dead hidden neurons dead.

    Kernel masks zero the incoming column *and* the outgoing row of every
    dead neuron — including rows feeding the output layer, so a later
    ``reconstruct`` physically removes the unit — and bias masks zero the
    dead units' biases, which would otherwise regrow during fine-tuning
    and resurrect the neuron as a constant output.

    Parameters:
    -----------
    model : Sequential
        The (unreconstructed) model whose hidden neurons were zeroed.
    dead_neurons : Dict[int, np.ndarray], optional
        Dead neuron indices keyed by hidden-layer index.

    Returns:
    --------
    Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]
        ``(kernel_masks, bias_masks)``, both keyed by the layer's index
        among ALL Dense layers (hidden + output); bias masks cover hidden
        layers only.
    """

    dense = _dense_layers(model)
    if len(dense) < 2:
        log_or_print(
            "Model must have at least two Dense layers (hidden + output).",
            level="error",
        )
        raise ValueError("Model must have at least two Dense layers (hidden + output).")

    n_hidden = len(dense) - 1
    alive: list[np.ndarray] = []
    for i in range(n_hidden):
        n_units = dense[i].get_weights()[0].shape[1]
        a = np.ones(n_units, dtype=bool)
        if dead_neurons:
            dead = np.unique(np.asarray(list(dead_neurons.get(i, [])), dtype=int))
            dead = dead[(dead >= 0) & (dead < n_units)]
            a[dead] = False
        alive.append(a)

    kernel_masks: dict[int, np.ndarray] = {}
    bias_masks: dict[int, np.ndarray] = {}
    prev_alive = np.ones(dense[0].get_weights()[0].shape[0], dtype=bool)
    for i in range(n_hidden):
        kernel_masks[i] = prev_alive[:, None] & alive[i][None, :]
        bias_masks[i] = alive[i].copy()
        prev_alive = alive[i]
    n_out = dense[-1].get_weights()[0].shape[1]
    kernel_masks[n_hidden] = prev_alive[:, None] & np.ones((1, n_out), dtype=bool)
    return kernel_masks, bias_masks


class MaskCallback(tf.keras.callbacks.Callback):
    """
    Re-apply kernel (and optionally bias) masks after every train batch.

    Generalizes ``importance_pruner.HardMaskCallback``: masks are keyed by
    the layer's index among ALL Dense layers (hidden + output), and
    optional bias masks let *neuron*-level pruning keep dead units dead
    during fine-tuning.

    Parameters:
    -----------
    model : Sequential
        The model being fine-tuned.
    kernel_masks : Dict[int, np.ndarray]
        Boolean (or 0/1) kernel masks keyed by Dense-layer index.
    bias_masks : Dict[int, np.ndarray], optional
        Boolean (or 0/1) bias masks keyed by Dense-layer index.
    """

    def __init__(
        self,
        model: Sequential,
        kernel_masks: dict[int, np.ndarray],
        bias_masks: dict[int, np.ndarray] | None = None,
    ):
        super().__init__()
        self._dense = _dense_layers(model)
        self._kernel_masks: dict[int, np.ndarray] = {}
        self._bias_masks: dict[int, np.ndarray] = {}

        for i, mask in kernel_masks.items():
            if not 0 <= i < len(self._dense):
                log_or_print(
                    f"kernel mask index {i} outside 0..{len(self._dense) - 1}.",
                    level="error",
                )
                raise ValueError(
                    f"kernel mask index {i} outside 0..{len(self._dense) - 1}."
                )
            W = self._dense[i].get_weights()[0]
            m = np.asarray(mask)
            if m.shape != W.shape:
                log_or_print(
                    f"kernel_masks[{i}] shape {m.shape} != layer kernel shape {W.shape}.",
                    level="error",
                )
                raise ValueError(
                    f"kernel_masks[{i}] shape {m.shape} != layer kernel shape {W.shape}."
                )
            self._kernel_masks[i] = m.astype(W.dtype)

        for i, mask in (bias_masks or {}).items():
            if not 0 <= i < len(self._dense):
                log_or_print(
                    f"bias mask index {i} outside 0..{len(self._dense) - 1}.",
                    level="error",
                )
                raise ValueError(
                    f"bias mask index {i} outside 0..{len(self._dense) - 1}."
                )
            ws = self._dense[i].get_weights()
            if len(ws) < 2:
                log_or_print(f"Dense layer {i} has no bias to mask.", level="error")
                raise ValueError(f"Dense layer {i} has no bias to mask.")
            b = ws[1]
            m = np.asarray(mask)
            if m.shape != b.shape:
                log_or_print(
                    f"bias_masks[{i}] shape {m.shape} != layer bias shape {b.shape}.",
                    level="error",
                )
                raise ValueError(
                    f"bias_masks[{i}] shape {m.shape} != layer bias shape {b.shape}."
                )
            self._bias_masks[i] = m.astype(b.dtype)

    def on_train_batch_end(self, batch, logs=None):
        for i, layer in enumerate(self._dense):
            if i not in self._kernel_masks:
                continue
            ws = layer.get_weights()
            W = ws[0] * self._kernel_masks[i]
            if len(ws) > 1 and i in self._bias_masks:
                layer.set_weights([W, ws[1] * self._bias_masks[i]])
            else:
                layer.set_weights([W] + list(ws[1:]))


@dataclass
class IterativePruneResult:
    """
    Output of ``iterative_prune``.

    Attributes:
    -----------
    model : Sequential
        Final model — reconstructed (physically narrower) when
        ``reconstruct_model`` was True and reconstruction succeeded,
        otherwise the masked working model.
    history : List[Dict[str, Any]]
        One record per round (round 0 = post-pretrain baseline) with the
        planned fraction, removal counts, cumulative sparsity, and any
        validation metrics.  Sparsity values are fractions in ``[0, 1]``.
    n_rounds_executed : int
        Rounds whose cut was applied and kept (a rolled-back round does
        not count).
    rolled_back : bool
        Whether the run stopped early by restoring the best state.
    final_edge_sparsity : float
        Measured fraction of zeroed hidden-kernel entries of the working
        model before reconstruction.
    final_neuron_sparsity : float or None
        Fraction of hidden neurons removed (``"neurons"`` granularity
        only, else ``None``).
    keep_masks : Dict[int, np.ndarray] or None
        Cumulative edge keep-masks per hidden layer (``"edges"``
        granularity only, else ``None``).
    dead_neurons : Dict[int, np.ndarray] or None
        Cumulative dead neuron indices per hidden layer (``"neurons"``
        granularity only, else ``None``).
    """

    model: Sequential
    history: list[dict[str, Any]] = field(default_factory=list)
    n_rounds_executed: int = 0
    rolled_back: bool = False
    final_edge_sparsity: float = 0.0
    final_neuron_sparsity: float | None = None
    keep_masks: dict[int, np.ndarray] | None = None
    dead_neurons: dict[int, np.ndarray] | None = None

    def history_df(self) -> pd.DataFrame:
        """
        Per-round history as a DataFrame (one row per round).

        Returns:
        --------
        pd.DataFrame
            ``self.history`` rendered as a DataFrame.
        """

        return pd.DataFrame(self.history)


def iterative_prune(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_sparsity: float = 0.8,
    n_rounds: int = 8,
    schedule: Literal["linear", "cubic"] = "cubic",
    granularity: Literal["edges", "neurons"] = "edges",
    pretrain_epochs: int = 65,
    fine_tune_epochs: int = 3,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    fitness_kw: dict[str, Any] | None = None,
    hard_masking: bool = True,
    rollback_threshold: float | None = None,
    rollback_metric: str = "loss",
    rollback_mode: Literal["min", "max", "auto"] = "auto",
    reconstruct_model: bool = True,
    compile_kwargs: dict[str, object] | None = None,
    fit_verbose: int = 0,
    fit_kwargs: dict[str, Any] | None = None,
) -> IterativePruneResult:
    """
    Gradually prune a compiled Sequential model over multiple rounds.

    Each round scores the *current* subnetwork on the most recent
    ``HiddenWeightTracker`` snapshot window (round 1 uses the pre-training
    window; later rounds use the previous round's fine-tuning window, i.e.
    the compensating subnetwork), removes a scheduled fraction of the
    *surviving* structure, then fine-tunes under hard masks, which also
    produces the next snapshot window.

    Parameters:
    -----------
    model : Sequential
        **Compiled** model to prune (never modified; a working clone is
        used).
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    target_sparsity : float, optional
        Cumulative sparsity to reach, by default 0.8.
    n_rounds : int, optional
        Maximum number of cut rounds ``R``, by default 8.
    schedule : str, optional
        ``"linear"`` or ``"cubic"`` cumulative-sparsity schedule (see
        ``sparsity_schedule``), by default ``"cubic"``.
    granularity : str, optional
        ``"edges"`` (unstructured, kernel-mask based) or ``"neurons"``
        (structured, ``prune_neurons`` based), by default ``"edges"``.
    pretrain_epochs : int, optional
        Epochs of the initial training window, by default 65.
    fine_tune_epochs : int, optional
        Epochs of each round's fine-tuning window (>= 2 recommended so the
        plasticity and stability terms do not degenerate to zero), by
        default 3.
    validation_data : Tuple[np.ndarray, np.ndarray], optional
        ``(X_val, y_val)`` used for per-round metrics, rollback checks and
        Keras ``validation_data``.
    fitness_kw : Dict[str, Any], optional
        Extra kwargs forwarded to ``compute_all_importance``.
    hard_masking : bool, optional
        Enforce the cumulative masks during every fine-tuning window.  If
        False, pruned structure may regrow within a window but is
        re-zeroed at the next round's cut, and ``reconstruct`` may keep
        regrown units, by default True.
    rollback_threshold : float or None, optional
        Stop early and restore the best state when ``rollback_metric``
        degrades by more than this much relative to its best value so far
        (``None`` disables rollback), by default None.
    rollback_metric : str, optional
        Metric name from ``model.evaluate`` used for rollback (e.g.
        ``"loss"`` or ``"accuracy"``), by default ``"loss"``.
    rollback_mode : str, optional
        ``"min"`` (lower is better), ``"max"``, or ``"auto"`` (``"min"``
        iff the metric name contains ``"loss"``), by default ``"auto"``.
    reconstruct_model : bool, optional
        Physically remove dead structure via ``importance_pruner.reconstruct``
        at the end, by default True.
    compile_kwargs : Dict[str, object], optional
        Compile arguments for the working clone; if ``None``, the original
        model's compile config is re-used.
    fit_verbose : int, optional
        Keras ``verbose`` for all ``fit`` calls, by default 0.
    fit_kwargs : Dict[str, Any], optional
        Extra kwargs forwarded to every ``fit`` call (e.g.
        ``batch_size``); must not contain ``epochs``, ``verbose``,
        ``callbacks`` or ``validation_data``.

    Returns:
    --------
    IterativePruneResult
        The final model, per-round history, cumulative masks / dead sets,
        and summary statistics.

    Raises:
    -------
    ValueError
        On invalid parameter combinations or an unknown ``rollback_metric``.
    """

    if granularity not in ("edges", "neurons"):
        log_or_print(
            f"granularity must be 'edges' or 'neurons', got '{granularity}'.",
            level="error",
        )
        raise ValueError(f"granularity must be 'edges' or 'neurons', got '{granularity}'.")
    if pretrain_epochs < 1:
        log_or_print(f"pretrain_epochs must be >= 1, got {pretrain_epochs}.", level="error")
        raise ValueError(f"pretrain_epochs must be >= 1, got {pretrain_epochs}.")
    if fine_tune_epochs < 1:
        log_or_print(
            f"fine_tune_epochs must be >= 1, got {fine_tune_epochs}.", level="error"
        )
        raise ValueError(f"fine_tune_epochs must be >= 1, got {fine_tune_epochs}.")
    if pretrain_epochs < 2 or fine_tune_epochs < 2:
        log_or_print(
            "pretrain_epochs / fine_tune_epochs < 2: single-epoch snapshot "
            "windows make the plasticity and stability terms of "
            "temporal_importance identically zero.",
            level="warning",
        )
    if rollback_threshold is not None and validation_data is None:
        log_or_print(
            "rollback_threshold given without validation_data; rollback disabled.",
            level="warning",
        )
        rollback_threshold = None

    targets = sparsity_schedule(target_sparsity, n_rounds, schedule)
    fitness_kw = dict(fitness_kw or {})
    fit_kwargs = dict(fit_kwargs or {})

    work = clone_model(model)
    work.set_weights(model.get_weights())
    if compile_kwargs is not None:
        work.compile(**compile_kwargs)
    else:
        work.compile(**restorable_compile_config(model))

    dense = _dense_layers(work)
    hidden = dense[:-1]
    n_hidden = len(hidden)
    if n_hidden < 1:
        log_or_print("Model needs at least one hidden Dense layer.", level="error")
        raise ValueError("Model needs at least one hidden Dense layer.")

    val_X, val_y = validation_data if validation_data is not None else (None, None)

    tracker = HiddenWeightTracker(record_epochs=pretrain_epochs)
    work.fit(
        X,
        y,
        epochs=pretrain_epochs,
        validation_data=validation_data,
        verbose=fit_verbose,
        callbacks=[tracker],
        **fit_kwargs,
    )

    baseline_metrics = _evaluate_metrics(work, val_X, val_y) if val_X is not None else None
    if (
        baseline_metrics is not None
        and rollback_threshold is not None
        and rollback_metric not in baseline_metrics
    ):
        log_or_print(
            f"rollback_metric '{rollback_metric}' not among model metrics "
            f"{sorted(baseline_metrics)}.",
            level="error",
        )
        raise ValueError(
            f"rollback_metric '{rollback_metric}' not among model metrics "
            f"{sorted(baseline_metrics)}."
        )

    history: list[dict[str, Any]] = []
    record: dict[str, Any] = {"round": 0, "phase": "pretrain", "edge_sparsity": 0.0}
    if granularity == "neurons":
        record["neuron_sparsity"] = 0.0
    if baseline_metrics is not None:
        record.update(_rounded(baseline_metrics))
    history.append(record)

    alive: dict[int, np.ndarray] | None = None
    dead_neurons: dict[int, np.ndarray] | None = None
    if granularity == "edges":
        alive = {
            i: np.ones(hidden[i].get_weights()[0].shape, dtype=bool)
            for i in range(n_hidden)
        }
    else:
        dead_neurons = {i: np.asarray([], dtype=int) for i in range(n_hidden)}

    metric_mode = _resolve_metric_mode(rollback_metric, rollback_mode)
    best_value: float | None = None
    best_state: tuple[list, dict[int, np.ndarray] | None] | None = None
    if rollback_threshold is not None and baseline_metrics is not None:
        best_value = baseline_metrics[rollback_metric]
        best_state = _snapshot_state(work, alive, dead_neurons)

    total_units = sum(int(hidden[i].get_weights()[0].shape[1]) for i in range(n_hidden))
    n_rounds_executed = 0
    rolled_back = False

    for r in range(1, n_rounds + 1):
        if granularity == "edges" and alive is not None:
            current_sparsity = _alive_edge_sparsity(alive)
        elif dead_neurons is not None:
            dead_total = sum(len(d) for d in dead_neurons.values())
            current_sparsity = dead_total / total_units if total_units > 0 else 0.0
        else:
            current_sparsity = 0.0

        fraction = survivor_fraction(current_sparsity, float(targets[r - 1]))
        if fraction <= 0.0:
            log_or_print(
                f"Round {r}: cumulative sparsity target already reached "
                f"({current_sparsity:.3f}); stopping early."
            )
            break

        fitness = compute_all_importance(tracker.snapshots, **fitness_kw)

        record = {
            "round": r,
            "granularity": granularity,
            "fraction": round(fraction, 4),
            "target_sparsity": round(float(targets[r - 1]), 4),
        }

        if granularity == "edges" and alive is not None:
            edge_importance = compute_edge_importance(tracker.snapshots, fitness)
            keep = cut_surviving_edges(edge_importance, alive, fraction)
            removed = sum(
                int(np.count_nonzero(alive[i] & ~keep[i])) for i in range(n_hidden)
            )
            if removed == 0:
                log_or_print(
                    f"Round {r}: no edges below threshold (tied scores); stopping early.",
                    level="warning",
                )
                record["edges_removed"] = 0
                record["edge_sparsity"] = round(current_sparsity, 4)
                history.append(record)
                break
            for i in range(n_hidden):
                W, b = hidden[i].get_weights()
                hidden[i].set_weights([W * keep[i].astype(W.dtype), b])
                alive[i] = keep[i]
            record["edges_removed"] = int(removed)
            record["edge_sparsity"] = round(_alive_edge_sparsity(alive), 4)
            log_or_print(
                f"Iterative round {r}/{n_rounds}: removed {removed} edges "
                f"(cumulative sparsity {record['edge_sparsity']:.3f})."
            )
        elif dead_neurons is not None:
            newly_dead = {
                i: select_weak_live(fitness[i], dead_neurons[i], fraction)
                for i in range(n_hidden)
            }
            removed = sum(int(len(w)) for w in newly_dead.values())
            if removed == 0:
                log_or_print(
                    f"Round {r}: no live neurons selected for removal; stopping early.",
                    level="warning",
                )
                record["neurons_removed"] = 0
                record["neuron_sparsity"] = round(current_sparsity, 4)
                history.append(record)
                break
            for i in range(n_hidden):
                dead_neurons[i] = np.union1d(dead_neurons[i], newly_dead[i])
            prune_neurons(
                work, {hidden[i].name: dead_neurons[i] for i in range(n_hidden)}
            )
            dead_total = sum(len(d) for d in dead_neurons.values())
            record["neurons_removed"] = int(removed)
            record["neuron_sparsity"] = round(
                dead_total / total_units if total_units > 0 else 0.0, 4
            )
            log_or_print(
                f"Iterative round {r}/{n_rounds}: removed {removed} neurons "
                f"(cumulative sparsity {record['neuron_sparsity']:.3f})."
            )

        tracker = HiddenWeightTracker()
        callbacks: list[tf.keras.callbacks.Callback] = [tracker]
        if hard_masking:
            if granularity == "edges" and alive is not None:
                kernel_masks, bias_masks = (
                    {i: alive[i] for i in range(n_hidden)},
                    None,
                )
            elif dead_neurons is not None:
                kernel_masks, bias_masks = neuron_masks(work, dead_neurons)
            else:
                kernel_masks, bias_masks = {}, None
            callbacks.append(MaskCallback(work, kernel_masks, bias_masks))
        work.fit(
            X,
            y,
            epochs=fine_tune_epochs,
            validation_data=validation_data,
            verbose=fit_verbose,
            callbacks=callbacks,
            **fit_kwargs,
        )
        n_rounds_executed = r

        if baseline_metrics is not None:
            metrics = _evaluate_metrics(work, val_X, val_y)
            record.update(_rounded(metrics))
            history.append(record)
            if (
                rollback_threshold is not None
                and best_value is not None
                and best_state is not None
            ):
                value = metrics[rollback_metric]
                if _is_better(value, best_value, metric_mode):
                    best_value = value
                    best_state = _snapshot_state(work, alive, dead_neurons)
                elif _degraded(value, best_value, metric_mode, rollback_threshold):
                    log_or_print(
                        f"Round {r}: validation '{rollback_metric}' degraded by "
                        f"more than {rollback_threshold}; rolling back to the "
                        f"best state and stopping."
                    )
                    work.set_weights(best_state[0])
                    if granularity == "edges":
                        alive = copy.deepcopy(best_state[1])
                    else:
                        dead_neurons = copy.deepcopy(best_state[1])
                    n_rounds_executed = r - 1
                    rolled_back = True
                    break
        else:
            history.append(record)

    final_edge_sparsity = _model_edge_sparsity(work)
    final_neuron_sparsity: float | None = None
    if granularity == "neurons" and dead_neurons is not None:
        dead_total = sum(len(d) for d in dead_neurons.values())
        final_neuron_sparsity = dead_total / total_units if total_units > 0 else 0.0

    final_model = work
    if reconstruct_model:
        try:
            final_model = reconstruct(work)
        except (RuntimeError, ValueError) as exc:
            log_or_print(
                f"reconstruct() failed ({exc}); returning the unreconstructed model.",
                level="warning",
            )

    result = IterativePruneResult(
        model=final_model,
        history=history,
        n_rounds_executed=n_rounds_executed,
        rolled_back=rolled_back,
        final_edge_sparsity=final_edge_sparsity,
        final_neuron_sparsity=final_neuron_sparsity,
        keep_masks=copy.deepcopy(alive) if granularity == "edges" else None,
        dead_neurons=copy.deepcopy(dead_neurons) if granularity == "neurons" else None,
    )

    summary = (
        f"Iterative pruning done: {n_rounds_executed} round(s), "
        f"edge sparsity {final_edge_sparsity:.3f}"
    )
    if final_neuron_sparsity is not None:
        summary += f", neuron sparsity {final_neuron_sparsity:.3f}"
    if rolled_back:
        summary += ", rolled back to best state"
    log_or_print(summary + ".")

    return result


def _dense_layers(model: Sequential) -> list[tf.keras.layers.Layer]:
    """
    All Dense layers with weights (hidden + output), in model order.

    Parameters:
    -----------
    model : Sequential
        The Keras model to extract Dense layers from.

    Returns:
    --------
    List[tf.keras.layers.Layer]
        The Dense layers.
    """

    return [l for l in model.layers if isinstance(l, Dense) and l.get_weights()]


def _evaluate_metrics(
    model: Sequential, X: np.ndarray | None, y: np.ndarray | None
) -> dict[str, float]:
    """
    Evaluate the model and return ``{metric_name: value}``.

    Parameters:
    -----------
    model : Sequential
        The compiled model to evaluate.
    X : np.ndarray or None
        Validation features.
    y : np.ndarray or None
        Validation labels.

    Returns:
    --------
    Dict[str, float]
        Loss and compiled metrics, keyed by ``model.metrics_names``.
    """

    values = np.atleast_1d(model.evaluate(X, y, verbose=0))
    return {name: float(v) for name, v in zip(model.metrics_names, values)}


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    """
    Round every metric value to 4 decimals.

    Parameters:
    -----------
    metrics : Dict[str, float]
        Metrics to round.

    Returns:
    --------
    Dict[str, float]
        The rounded metrics.
    """

    return {k: round(float(v), 4) for k, v in metrics.items()}


def _resolve_metric_mode(
    metric: str, mode: Literal["min", "max", "auto"]
) -> Literal["min", "max"]:
    """
    Resolve the rollback direction of a metric.

    Parameters:
    -----------
    metric : str
        Metric name.
    mode : str
        ``"min"``, ``"max"`` or ``"auto"``.

    Returns:
    --------
    str
        ``"min"`` if lower is better, else ``"max"``.
    """

    if mode in ("min", "max"):
        return mode
    return "min" if "loss" in metric.lower() else "max"


def _is_better(new: float, best: float, mode: Literal["min", "max"]) -> bool:
    """
    Whether ``new`` improves on ``best`` under the given direction.

    Parameters:
    -----------
    new : float
        New metric value.
    best : float
        Best metric value so far.
    mode : str
        ``"min"`` or ``"max"``.

    Returns:
    --------
    bool
        True if ``new`` is the better value.
    """

    return new < best if mode == "min" else new > best


def _degraded(
    new: float, best: float, mode: Literal["min", "max"], threshold: float
) -> bool:
    """
    Whether ``new`` is worse than ``best`` by more than ``threshold``.

    Parameters:
    -----------
    new : float
        New metric value.
    best : float
        Best metric value so far.
    mode : str
        ``"min"`` or ``"max"``.
    threshold : float
        Allowed degradation.

    Returns:
    --------
    bool
        True if the degradation exceeds ``threshold``.
    """

    # return (new - best) > threshold if mode == "min" else (best - new) > threshold
    return (new - best) / abs(best) > threshold if mode == "min" else (best - new) / abs(best) > threshold


def _snapshot_state(
    model: Sequential,
    alive: dict[int, np.ndarray] | None,
    dead_neurons: dict[int, np.ndarray] | None,
) -> tuple[list, dict[int, np.ndarray] | None]:
    """
    Deep copy of the weights plus the pruning state (alive masks or dead sets).

    Parameters:
    -----------
    model : Sequential
        Model to snapshot.
    alive : Dict[int, np.ndarray] or None
        Cumulative edge keep-masks (edge granularity).
    dead_neurons : Dict[int, np.ndarray] or None
        Cumulative dead neuron sets (neuron granularity).

    Returns:
    --------
    Tuple[List, Dict[int, np.ndarray] or None]
        ``(weights, state)`` deep copies.
    """

    state = alive if alive is not None else dead_neurons
    return copy.deepcopy(model.get_weights()), copy.deepcopy(state)


def _alive_edge_sparsity(alive: dict[int, np.ndarray]) -> float:
    """
    Edge-count-weighted dead fraction over all hidden-layer keep-masks.

    Matches the ``survivor_fraction`` accounting, which treats sparsity as
    a fraction of *all* hidden-kernel edges, not a mean of per-layer
    fractions.

    Parameters:
    -----------
    alive : Dict[int, np.ndarray]
        Cumulative boolean keep-masks per hidden layer.

    Returns:
    --------
    float
        Cumulative edge sparsity in ``[0, 1]``.
    """

    total = sum(int(a.size) for a in alive.values())
    live = sum(int(np.count_nonzero(a)) for a in alive.values())
    return 1.0 - (live / total) if total > 0 else 0.0


def _model_edge_sparsity(model: Sequential) -> float:
    """
    Measured fraction of zeroed entries across hidden-kernel weights.

    Parameters:
    -----------
    model : Sequential
        The model to measure.

    Returns:
    --------
    float
        Zero-entry fraction in ``[0, 1]``.
    """

    total = 0
    zeros = 0
    for layer in _dense_layers(model)[:-1]:
        W = layer.get_weights()[0]
        total += W.size
        zeros += int(np.sum(W == 0))
    return zeros / total if total > 0 else 0.0
