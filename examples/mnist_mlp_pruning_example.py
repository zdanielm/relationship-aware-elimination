import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Sequential
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

# --- MNIST MLP model config ---
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
            Input(shape=(input_size,)),
            Dense(
                HIDDEN_UNITS,
                activation="relu",
                kernel_regularizer=l2(0.001),
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


def confusion_diff_plot(y_true, y_pred_pre, y_pred_post, num_labels, title, out_path):
    cm_pre = confusion_matrix(y_true, y_pred_pre, labels=range(num_labels))
    cm_post = confusion_matrix(y_true, y_pred_post, labels=range(num_labels))
    diff = cm_post - cm_pre

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = int(np.abs(diff).max()) or 1
    im = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="count change (final - pretrain)")
    ax.set_xticks(range(num_labels))
    ax.set_yticks(range(num_labels))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(num_labels):
        for j in range(num_labels):
            if diff[i, j]:
                ax.text(j, i, str(int(diff[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(seed):
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)

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
    y_pred_pre = np.argmax(model.predict(x_test, verbose=0), axis=1)

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
    results.insert(0, "seed", seed)
    out_path = "mnist_mlp_pruning_results.csv"
    results.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"\nSaved all results to {out_path}")

    y_pred_final = np.argmax(result.model.predict(x_test, verbose=0), axis=1)
    diff_path = f"mnist_confusion_diff_seed{seed}.png"
    confusion_diff_plot(
        y_test_int,
        y_pred_pre,
        y_pred_final,
        num_labels,
        f"Confusion matrix difference: final pruned ({iterative_post['sparsity_pct']}% sparse) - pretrain",
        diff_path,
    )
    print(f"Saved confusion matrix difference to {diff_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args().seed)
