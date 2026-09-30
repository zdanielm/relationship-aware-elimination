"""
Graph-theoretic neuron pruning using Netork Science-based importance metrics.

Provides utilities that work on a Keras Sequential model with an
arbitrary number of hidden Dense layers.

Input and output layers are kept intact.

Main entry points:
------------------
+ ``HiddenWeightTracker``  – Keras callback that tracks hidden-layer
  kernels at every (or selected) epoch(s).
+ ``temporal_metrics``     – builds per-epoch strength, concentration
  (max/sum on incident weights), and centrality for every hidden neuron.
+ ``temporal_importance`` – collapses the temporal series into one
  scalar importance per hidden neuron (optional per-layer min–max of each
  term so weights are comparable).
+ ``select_weak``          – picks the bottom-k neurons by importance.
+ ``prune_neurons``        – zeros out all connections of chosen neurons
  (structured, shape-preserving).
+ ``prune_edges_unstructured`` – zeros individual weights (edges) whose
  importance-weighted importance is lowest (unstructured).
+ ``HardMaskCallback`` – optional callback to keep pruned edges at zero during
  fine-tuning (``hard_masking`` in ``prune_edges_unstructured``).
+ ``reconstruct``          – builds a new, *narrower* model that
  physically removes the pruned neurons, copying surviving weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, clone_model

from utils.logging_utils import log_or_print

_CENTRALITY_METRIC_KEYS = frozenset({"ec_weight", "dc_weight", "bc_weight"})


class HiddenWeightTracker(tf.keras.callbacks.Callback):
    """Records hidden-layer kernel snapshots during training.

    Parameters:
    -----------
    record_epochs : int or None
        If given, only the first *record_epochs* epochs are recorded.
        ``None`` records every epoch.
    """

    def __init__(self, record_epochs: Optional[int] = None):
        super().__init__()
        self.record_epochs = record_epochs
        self.snapshots: List[List[np.ndarray]] = []

    def on_epoch_end(self, epoch, logs=None):
        if self.record_epochs is not None and epoch >= self.record_epochs:
            return
        hidden = _hidden_dense_layers(self.model)
        self.snapshots.append([l.get_weights()[0].copy() for l in hidden])


def _hidden_dense_layers(model: Sequential) -> List[tf.keras.layers.Layer]:
    """
    Return only the hidden Dense layers (skip input & output).

    Parameters:
    -----------
    model : Sequential
        The Keras model to extract hidden Dense layers from.

    Returns:
    --------
    List[tf.keras.layers.Layer]
        The hidden Dense layers.

    Raises:
    -------
    ValueError
        If the model does not have at least two Dense layers (hidden + output).
    """

    dense_layers = [l for l in model.layers if isinstance(l, Dense) and l.get_weights()]
    if len(dense_layers) < 2:
        log_or_print(
            "Model must have at least two Dense layers (hidden + output).",
            level="error",
        )
        raise ValueError("Model must have at least two Dense layers (hidden + output).")
    return dense_layers[:-1]


def _hidden_layer_names(model: Sequential) -> List[str]:
    """
    Return the names of the hidden Dense layers (skip input & output).

    Parameters:
    -----------
    model : Sequential
        The Keras model to extract hidden Dense layer names from.

    Returns:
    --------
    List[str]
        The names of the hidden Dense layers.
    """

    return [l.name for l in _hidden_dense_layers(model)]


def build_digraph(kernels: Sequence[np.ndarray]) -> nx.DiGraph:
    """Build a weighted DiGraph from a list of weight matrices.

    Parameters:
    -----------
    kernels : Sequence[np.ndarray]
        A sequence of weight matrices, one for each layer.

    Returns:
    --------
    nx.DiGraph
        The weighted DiGraph representing the model's connectivity.
    """

    G = nx.DiGraph()
    prev_prefix = "in"
    for layer_idx, W in enumerate(kernels):
        cur_prefix = f"h{layer_idx}"
        n_src, n_dst = W.shape
        for i in range(n_src):
            for j in range(n_dst):
                w = abs(float(W[i, j]))
                if w > 0:
                    G.add_edge(f"{prev_prefix}_{i}", f"{cur_prefix}_{j}", weight=w)
        prev_prefix = cur_prefix
    return G


def _minmax(x: np.ndarray) -> np.ndarray:
    """
    Min–max normalizes the array x to the [0, 1] interval.

    If all elements are equal (max == min), returns an array of zeros
    with the same shape as x.

    Parameters:
    -----------
    x : np.ndarray
        Input array to normalize.

    Returns:
    --------
    np.ndarray
        The normalized array, with values scaled to [0, 1].
    """

    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _incident_abs_weights(G: nx.DiGraph, node: str) -> List[float]:
    """
    Absolute weights on all edges incident to ``node`` (in + out).

    Parameters:
    -----------
    G : nx.DiGraph
        The graph to query.
    node : str
        The node to find incident edges for.

    Returns:
    --------
    List[float]
        The absolute weights of all incident edges.
    """

    if node not in G:
        return []
    ws: List[float] = []
    for u in G.predecessors(node):
        ws.append(abs(float(G[u][node]["weight"])))
    for v in G.successors(node):
        ws.append(abs(float(G[node][v]["weight"])))
    return ws


def _strength_and_concentration(
    G: nx.DiGraph, prefix: str, j: int
) -> Tuple[float, float]:
    """
    Total incident absolute weight (strength) and max/sum concentration.

    Parameters:
    -----------
    G : nx.DiGraph
        The graph to query.
    prefix : str
        The node prefix (e.g., "in" or "hL").
    j : int
        The node index.

    Returns:
    --------
    Tuple[float, float]
        The total incident weight and max/sum concentration.
    """

    node = f"{prefix}_{j}"
    ws = _incident_abs_weights(G, node)
    if not ws:
        return 0.0, 0.0
    total = float(sum(ws))
    mx = float(max(ws))
    rho = mx / total if total > 0.0 else 0.0
    return total, rho


def temporal_metrics(
    snapshots: List[List[np.ndarray]],
    ec_weight: float = 0.50,
    dc_weight: float = 0.25,
    bc_weight: float = 0.25,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes per-hidden-layer strength, concentration, and centrality trajectories.

    Parameters:
    -----------
    snapshots : list[list[ndarray]]
        Outer list = epochs, inner list = kernel per hidden layer.
    ec_weight, dc_weight, bc_weight :
        Mixing coefficients for eigenvector / degree / betweenness
        centrality (must sum to 1).

    Returns:
    --------
    dict mapping *hidden-layer index* -> (strengths, centralities, concentrations)
        Each value is a triple of ``(T, n_units)`` arrays.  **concentrations**
        holds the max-to-sum ratio of absolute incident edge weights per
        neuron per epoch (0 if no incident edges).
    """

    n_hidden = len(snapshots[0])
    per_layer: Dict[int, Tuple[list, list, list]] = {
        i: ([], [], []) for i in range(n_hidden)
    }

    for epoch_kernels in snapshots:
        G = build_digraph(epoch_kernels)

        for layer_idx in range(n_hidden):
            prefix = f"h{layer_idx}"
            n_units = epoch_kernels[layer_idx].shape[1]

            strength_rho = [
                _strength_and_concentration(G, prefix, j) for j in range(n_units)
            ]
            strength = np.array([sr[0] for sr in strength_rho], dtype=float)
            concentration = np.array([sr[1] for sr in strength_rho], dtype=float)
            per_layer[layer_idx][0].append(strength)
            per_layer[layer_idx][2].append(concentration)

            ec = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight="weight")
            dc = nx.degree_centrality(G)
            bc = nx.betweenness_centrality(G, weight="weight")

            ec_vals = np.array([ec.get(f"{prefix}_{j}", 0.0) for j in range(n_units)])
            dc_vals = np.array([dc.get(f"{prefix}_{j}", 0.0) for j in range(n_units)])
            bc_vals = np.array([bc.get(f"{prefix}_{j}", 0.0) for j in range(n_units)])

            combined = (
                ec_weight * _minmax(ec_vals)
                + dc_weight * _minmax(dc_vals)
                + bc_weight * _minmax(bc_vals)
            )
            per_layer[layer_idx][1].append(combined)

    return {
        idx: (np.array(s), np.array(c), np.array(z))
        for idx, (s, c, z) in per_layer.items()
    }


def temporal_importance(
    strengths: np.ndarray,
    centralities: np.ndarray,
    concentrations: Optional[np.ndarray] = None,
    w_strength: float = 0.2,
    w_plasticity: float = 0.3,
    w_centrality: float = 0.2,
    w_stability: float = 0.1,
    w_concentration: float = 0.2,
    normalize: bool = True,
) -> np.ndarray:
    """
    Collapse temporal trajectories into one importance per neuron.

    Parameters:
    -----------
    strengths : np.ndarray
        Per-epoch strength trajectory (``T`` x ``n_units``).
    centralities : np.ndarray
        Per-epoch centrality trajectory (``T`` x ``n_units``).
    concentrations : Optional[np.ndarray], optional
        Per-epoch concentration trajectory (``T`` x ``n_units``), by default None.
    w_strength : float, optional
        Weight for strength term, by default 0.2.
    w_plasticity : float, optional
        Weight for plasticity term, by default 0.3.
    w_centrality : float, optional
        Weight for centrality term, by default 0.2.
    w_stability : float, optional
        Weight for stability term, by default 0.1.
    w_concentration : float, optional
        Weight for concentration term, by default 0.2.
    normalize : bool, optional
        Whether to normalize each feature vector to ``[0, 1]`` across neurons, by default True.

    Returns:
    --------
    np.ndarray
        1-D array of length ``n_units``.
    """

    if concentrations is None:
        concentrations = np.zeros_like(strengths)
    supra_strength = np.sum(strengths, axis=0)
    plasticity = np.sum(np.abs(np.diff(strengths, axis=0)), axis=0)
    total_centrality = np.sum(centralities, axis=0)
    stability = np.std(strengths, axis=0)
    mean_concentration = np.mean(concentrations, axis=0)
    if normalize:
        supra_strength = _minmax(supra_strength)
        plasticity = _minmax(plasticity)
        total_centrality = _minmax(total_centrality)
        stability = _minmax(stability)
        mean_concentration = _minmax(mean_concentration)
    return (
        w_strength * supra_strength
        + w_plasticity * plasticity
        + w_centrality * total_centrality
        - w_stability * stability
        + w_concentration * mean_concentration
    )


def compute_all_importance(
    snapshots: List[List[np.ndarray]],
    **kwargs,
) -> Dict[int, np.ndarray]:
    """
    Computes importance for each hidden layer, returning ``{hidden_layer_idx: importance_vector}``.

    Parameters:
    -----------
    snapshots : List[List[np.ndarray]]
        List of snapshots, where each snapshot is a list of layer activations.
    **kwargs
        Keyword arguments for ``temporal_metrics`` and ``temporal_importance``.

    Returns:
    --------
    Dict[int, np.ndarray]
        Dictionary mapping hidden layer index to importance vector.
    """

    metric_kw = {k: v for k, v in kwargs.items() if k in _CENTRALITY_METRIC_KEYS}
    importance_kw = {
        k: v for k, v in kwargs.items() if k not in _CENTRALITY_METRIC_KEYS
    }
    metrics = temporal_metrics(snapshots, **metric_kw)
    return {
        idx: temporal_importance(s, c, conc, **importance_kw)
        for idx, (s, c, conc) in metrics.items()
    }


def select_weak(importance: np.ndarray, prune_ratio: float = 0.2) -> np.ndarray:
    """
    Return *indices* of the weakest ``prune_ratio`` fraction of neurons.

    Parameters:
    -----------
    importance : np.ndarray
        Importance values of neurons.
    prune_ratio : float, optional
        Fraction of weakest neurons to select, by default 0.2.

    Returns:
    --------
    np.ndarray
        Indices of the weakest neurons.
    """

    k = int(len(importance) * prune_ratio)
    return np.argsort(importance)[:k]


def prune_neurons(
    model: Sequential,
    layer_neuron_indices: Dict[str, np.ndarray],
) -> None:
    """
    Zero all incoming/outgoing weights of selected neurons per layer.

    Parameters:
    -----------
    model : Keras Sequential
    layer_neuron_indices : dict[layer_name -> array of neuron indices]
        For each hidden layer name, which neuron indices to kill.

    Returns:
    --------
    None
    """

    dense_layers = [
        layer_name
        for layer_name in model.layers
        if isinstance(layer_name, Dense) and layer_name.get_weights()
    ]

    for i, layer in enumerate(dense_layers[:-1]):
        if layer.name not in layer_neuron_indices:
            continue
        idx = layer_neuron_indices[layer.name]
        if len(idx) == 0:
            continue

        W_in, b = layer.get_weights()
        W_in[:, idx] = 0.0
        b[idx] = 0.0
        layer.set_weights([W_in, b])

        next_layer = dense_layers[i + 1]
        W_out, b_out = next_layer.get_weights()
        W_out[idx, :] = 0.0
        next_layer.set_weights([W_out, b_out])


def compute_edge_importance(
    snapshots: List[List[np.ndarray]],
    importance_dict: Dict[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Importance-weighted edge importance summed over epochs.

    For each hidden layer *L* with kernel shape ``(fan_in, fan_out)``:
        importance[i, j] = sum_t |W_L^t[i,j]| * importance_src[i] * importance_dst[j]

    Input-layer and output-layer neurons get a neutral importance of 1.0.

    Parameters:
    -----------
    snapshots : from ``HiddenWeightTracker.snapshots``
    importance_dict : ``{hidden_layer_idx: 1-D importance}`` from
        ``compute_all_importance``.

    Returns:
    --------
    Dict[str, np.ndarray]
        Edge importance for each hidden layer, with keys like ``'hidden_0'``.
    """

    n_hidden = len(snapshots[0])
    W_arrays = {i: np.array([ep[i] for ep in snapshots]) for i in range(n_hidden)}
    importance: Dict[str, np.ndarray] = {}

    for layer_idx in range(n_hidden):
        W = W_arrays[layer_idx]  # (T, fan_in, fan_out)
        src_fit = importance_dict.get(layer_idx - 1, None)  # previous hidden
        dst_fit = importance_dict.get(layer_idx, None)  # this hidden

        scaled = np.abs(W)
        if dst_fit is not None:
            scaled = scaled * dst_fit[np.newaxis, np.newaxis, :]
        if src_fit is not None:
            scaled = scaled * src_fit[np.newaxis, :, np.newaxis]

        importance[layer_idx] = np.sum(scaled, axis=0)

    return importance


class HardMaskCallback(tf.keras.callbacks.Callback):
    """
    Re-apply binary kernel masks after each train batch so pruned weights stay zero.
    """

    def __init__(self, model: Sequential, kernel_masks: Dict[int, np.ndarray]):
        super().__init__()
        self._hidden = _hidden_dense_layers(model)
        if len(self._hidden) != len(kernel_masks):
            log_or_print(
                f"kernel_masks has {len(kernel_masks)} layers; "
                f"model has {len(self._hidden)} hidden Dense layers.",
                level="error",
            )
            raise ValueError(
                f"kernel_masks has {len(kernel_masks)} layers; "
                f"model has {len(self._hidden)} hidden Dense layers."
            )
        self._kernel_masks: Dict[int, np.ndarray] = {}
        for i, layer in enumerate(self._hidden):
            W, _ = layer.get_weights()
            m = np.asarray(kernel_masks[i])
            if m.shape != W.shape:
                log_or_print(
                    f"kernel_masks[{i}] shape {m.shape} != layer kernel shape {W.shape}.",
                    level="error",
                )
                raise ValueError(
                    f"kernel_masks[{i}] shape {m.shape} != layer kernel shape {W.shape}."
                )
            self._kernel_masks[i] = m

    def on_train_batch_end(self, batch, logs=None):
        for i, layer in enumerate(self._hidden):
            W, b = layer.get_weights()
            m = self._kernel_masks[i]
            layer.set_weights([W * m, b])


def prune_edges_unstructured(
    model: Sequential,
    importance: Dict[int, np.ndarray],
    prune_ratio: float = 0.2,
    *,
    hard_masking: bool = False,
) -> Tuple[Sequential, int, int, float, Optional[Dict[int, np.ndarray]]]:
    """
    Zero the lowest-importance edges across all hidden layers, on a copy of the model.

    Parameters:
    -----------
    hard_masking
        If True, ``kernel_masks`` is returned in the 5th element so you can pass it to
        ``HardMaskCallback`` during fine-tuning so pruned weights cannot grow back.

    Returns:
    --------
    new_model, n_pruned, total, sparsity_percent, kernel_masks_or_none
    """

    # make a working copy of the model
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())
    hidden = _hidden_dense_layers(new_model)
    all_scores = np.concatenate([importance[i].ravel() for i in range(len(hidden))])
    total = len(all_scores)
    threshold = np.percentile(all_scores, prune_ratio * 100)

    kernel_masks: Optional[Dict[int, np.ndarray]] = {} if hard_masking else None
    n_pruned = 0
    for i, layer in enumerate(hidden):
        W, b = layer.get_weights()
        mask = (importance[i] >= threshold).astype(W.dtype)
        n_pruned += int(np.sum(mask == 0))
        W_pruned = W * mask
        layer.set_weights([W_pruned, b])
        if hard_masking and kernel_masks is not None:
            kernel_masks[i] = mask.copy()

    log_or_print(
        f"Unstructured edge pruning: removed {n_pruned}/{total} edges "
        f"({n_pruned / total * 100:.1f}%)"
    )
    sparsity_percent = round((100 * (n_pruned / total)), 2)
    return new_model, n_pruned, total, sparsity_percent, kernel_masks


def _surviving_mask(layer: tf.keras.layers.Layer) -> np.ndarray:
    """
    Boolean mask of neurons that still have at least one non-zero
    incoming *or* outgoing connection.

    Parameters:
    -----------
    layer : tf.keras.layers.Layer
        The layer to compute the mask for.

    Returns:
    --------
    np.ndarray
        Boolean mask of surviving neurons.
    """

    W = layer.get_weights()[0]
    col_alive = np.any(W != 0, axis=0)
    return col_alive


def reconstruct(
    model: Sequential,
    compile_kwargs: Optional[dict] = None,
) -> Sequential:
    """
    Return a new, narrower Sequential model without dead hidden neurons.

    Dead = every incoming weight **and** every outgoing weight is zero.
    Input and output dimensions are preserved.

    Parameters:
    -----------
    model : the pruned model.
    compile_kwargs : dict passed to ``new_model.compile()``.  If
        ``None``, the original model's optimizer, loss, and metric
        config are re-used.

    Returns:
    --------
    Sequential
        The reconstructed model.
    """

    dense_layers = [
        layer_name
        for layer_name in model.layers
        if isinstance(layer_name, Dense) and layer_name.get_weights()
    ]
    if len(dense_layers) < 2:
        log_or_print(
            "At least 2 Dense layers (hidden + output) are required.", level="error"
        )
        raise ValueError("At least 2 Dense layers (hidden + output) are required.")

    hidden = dense_layers[:-1]
    output_layer = dense_layers[-1]

    alive_masks: List[np.ndarray] = []
    for i, layer in enumerate(hidden):
        W_in = layer.get_weights()[0]
        col_alive = np.any(W_in != 0, axis=0)
        if i + 1 < len(dense_layers):
            W_out = dense_layers[i + 1].get_weights()[0]
            row_alive = np.any(W_out != 0, axis=1)
            alive = col_alive | row_alive
        else:
            alive = col_alive
        alive_masks.append(alive)

    new_model = Sequential()
    input_dim = hidden[0].get_weights()[0].shape[0]

    for i, (layer, alive) in enumerate(zip(hidden, alive_masks)):
        keep_idx = np.where(alive)[0]
        if len(keep_idx) == 0:
            raise RuntimeError(
                f"All neurons in hidden layer '{layer.name}' are dead — "
                "cannot reconstruct."
            )
        W_old, b_old = layer.get_weights()
        W_new = W_old[:, keep_idx]
        b_new = b_old[keep_idx]

        if i > 0:
            prev_keep = np.where(alive_masks[i - 1])[0]
            W_new = W_new[prev_keep, :]

        activation = layer.get_config().get("activation", "relu")
        kwargs = dict(units=len(keep_idx), activation=activation, name=f"dense_{i}")
        if i == 0:
            kwargs["input_shape"] = (input_dim,)
        new_layer = Dense(**kwargs)
        new_model.add(new_layer)
        new_model.layers[-1].build(
            (None, input_dim) if i == 0 else (None, int(np.sum(alive_masks[i - 1])))
        )
        new_model.layers[-1].set_weights([W_new, b_new])

    W_out_old, b_out_old = output_layer.get_weights()
    last_keep = np.where(alive_masks[-1])[0]
    W_out_new = W_out_old[last_keep, :]
    out_activation = output_layer.get_config().get("activation", "sigmoid")
    out_units = output_layer.get_config()["units"]
    new_model.add(Dense(out_units, activation=out_activation, name="output"))
    new_model.layers[-1].build((None, len(last_keep)))
    new_model.layers[-1].set_weights([W_out_new, b_out_old])

    if compile_kwargs is not None:
        new_model.compile(**compile_kwargs)
    else:
        cfg = model.get_compile_config()
        new_model.compile(**cfg)

    return new_model


def prune_edges_structured(
    model: Sequential,
    importance: Dict[int, np.ndarray],
    prune_ratio: float = 0.2,
) -> Tuple[Sequential, int, int, int]:
    """
    Prune edges in a structured manner, preserving the overall structure of the model.

    Parameters:
    -----------
    model : Sequential
        The model to prune.
    importance : Dict[int, np.ndarray]
        Edge importance scores for each layer.
    prune_ratio : float, optional
        Fraction of edges to prune (default is 0.2).

    Returns:
    --------
    Tuple[Sequential, int, int, int]
        The reconstructed model, number of pruned edges, total edges, and sparsity percentage.
    """

    zeroed_model, n_pruned_edges, total_edges, sparsity_percent, _ = (
        prune_edges_unstructured(model, importance, prune_ratio)
    )

    rebuilt_model = reconstruct(zeroed_model)
    return rebuilt_model, n_pruned_edges, total_edges, sparsity_percent
