import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

from src.mlp.keras.importance_pruner import (
    HardMaskCallback,
    HiddenWeightTracker,
    compute_all_importance,
    compute_edge_importance,
    prune_edges_unstructured,
)
from src.mlp.keras.iterative_pruner import iterative_prune

# --- MNIST MLP (same architecture as sandbox/MNIST_MLP.ipynb) ---
HIDDEN_UNITS = 64
BATCH_SIZE = 128
PRE_PRUNE_EPOCHS = 20
POST_PRUNE_EPOCHS = 10
PRUNE_RATIO = 0.5
TARGET_SPARSITY = 0.5
N_ROUNDS = 5
SEED = 1


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    return x_train, y_train, x_test, y_test


def build_model(input_size, num_labels):
    m = Sequential(
        [
            Dense(
                HIDDEN_UNITS,
                activation="relu",
                kernel_regularizer=l2(0.001),
                input_shape=(input_size,),
            ),
            Dense(
                HIDDEN_UNITS,
                activation="relu",
                kernel_regularizer=l2(0.001),
            ),
            Dense(num_labels, activation="softmax"),
        ]
    )
    m.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return m


def evaluate(model, x, y, label):
    results = model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
    total, zeros = 0, 0
    for layer in model.layers:
        ws = layer.get_weights()
        if ws:
            total += ws[0].size
            zeros += int(np.sum(ws[0] == 0))
    return {
        "model": label,
        "loss": round(results["loss"], 4),
        "accuracy": round(results["accuracy"], 4),
        "sparsity_pct": round(100 * zeros / total, 1) if total else 0.0,
    }


def main():
    np.random.seed(SEED)
    import tensorflow as tf

    tf.random.set_seed(SEED)

    x_train, y_train_int, x_test, y_test_int = load_mnist()
    num_labels = len(np.unique(y_train_int))
    y_train = tf.keras.utils.to_categorical(y_train_int, num_labels)
    y_test = tf.keras.utils.to_categorical(y_test_int, num_labels)

    # --- One-shot pruning (train -> score -> cut once -> fine-tune) ---
    model = build_model(x_train.shape[1], num_labels)
    model.fit(
        x_train,
        y_train,
        epochs=PRE_PRUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    oneshot_pre = evaluate(model, x_test, y_test, "oneshot_pre")

    tracker = HiddenWeightTracker()
    model.fit(
        x_train,
        y_train,
        epochs=PRE_PRUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[tracker],
    )
    fitness = compute_all_importance(tracker.snapshots)
    importance = compute_edge_importance(tracker.snapshots, fitness)
    pruned_model, n_pruned, total, _, kernel_masks = prune_edges_unstructured(
        model,
        importance,
        prune_ratio=PRUNE_RATIO,
        hard_masking=True,
    )
    pruned_model.fit(
        x_train,
        y_train,
        epochs=POST_PRUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[HardMaskCallback(pruned_model, kernel_masks)],
    )
    oneshot_post = evaluate(pruned_model, x_test, y_test, "oneshot_post")
    print(f"One-shot: pruned {n_pruned}/{total} hidden edges")
    print(pd.DataFrame([oneshot_pre, oneshot_post]).to_string(index=False))

    # --- Iterative pruning (score -> cut a fraction -> fine-tune, x N_ROUNDS) ---
    model_iter = build_model(x_train.shape[1], num_labels)
    result = iterative_prune(
        model_iter,
        x_train,
        y_train,
        target_sparsity=TARGET_SPARSITY,
        n_rounds=N_ROUNDS,
        schedule="cubic",
        granularity="edges",
        pretrain_epochs=PRE_PRUNE_EPOCHS,
        fine_tune_epochs=max(2, POST_PRUNE_EPOCHS // N_ROUNDS),
        validation_data=(x_test, y_test),
        hard_masking=True,
        reconstruct_model=True,
        fit_verbose=0,
        fit_kwargs={"batch_size": BATCH_SIZE},
    )
    iterative_post = evaluate(result.model, x_test, y_test, "iterative_post")
    print(f"\nIterative: {result.n_rounds_executed} round(s), rolled_back={result.rolled_back}")
    print(result.history_df().to_string(index=False))
    print(pd.DataFrame([iterative_post]).to_string(index=False))

    # --- Save everything to one CSV ---
    results = pd.concat(
        [
            pd.DataFrame([oneshot_pre, oneshot_post]),
            result.history_df(),
            pd.DataFrame([iterative_post]),
        ],
        ignore_index=True,
    )
    out_path = "mnist_mlp_pruning_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved all results to {out_path}")


if __name__ == "__main__":
    main()
