import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from ucimlrepo import fetch_ucirepo

from src.mlp.keras.importance_pruner import (
    HardMaskCallback,
    HiddenWeightTracker,
    compute_all_importance,
    compute_edge_importance,
    prune_edges_unstructured,
)
from src.mlp.keras.iterative_pruner import iterative_prune

# --- Sonar (Mines vs Rocks) MLP model config ---
BATCH_SIZE = 128
PRE_PRUNE_EPOCHS = 20
POST_PRUNE_EPOCHS = 10
PRUNE_RATIO = 0.5
TARGET_SPARSITY = 0.5
N_ROUNDS = 5
SEED = 1


def load_sonar():
    sonar = fetch_ucirepo(id=151)
    X = sonar.data.features
    y = sonar.data.targets["class"].map({"R": 0, "M": 1})
    return X, y


def build_model(input_size):
    m = Sequential(
        [
            Input(shape=(input_size,)),
            Dense(16, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    m.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="roc_auc"),
            AUC(name="pr_auc", curve="PR"),
        ],
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

    from sklearn.model_selection import train_test_split

    X, y = load_sonar()
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    num_labels = 2

    # --- One-shot pruning (train -> score -> cut once -> fine-tune) ---
    model = build_model(train_X.shape[1])
    model.fit(
        train_X,
        train_y,
        epochs=PRE_PRUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    oneshot_pre = evaluate(model, test_X, test_y, "oneshot_pre")
    y_pred_pre = (model.predict(test_X, verbose=0) > 0.5).astype(int).flatten()

    tracker = HiddenWeightTracker()
    model.fit(
        train_X,
        train_y,
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
        train_X,
        train_y,
        epochs=POST_PRUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[HardMaskCallback(pruned_model, kernel_masks)],
    )
    oneshot_post = evaluate(pruned_model, test_X, test_y, "oneshot_post")
    print(f"One-shot: pruned {n_pruned}/{total} hidden edges")
    print(pd.DataFrame([oneshot_pre, oneshot_post]).to_string(index=False))

    # --- Iterative pruning (score -> cut a fraction -> fine-tune, x N_ROUNDS) ---
    model_iter = build_model(train_X.shape[1])
    result = iterative_prune(
        model_iter,
        train_X,
        train_y,
        target_sparsity=TARGET_SPARSITY,
        n_rounds=N_ROUNDS,
        schedule="cubic",
        granularity="edges",
        pretrain_epochs=PRE_PRUNE_EPOCHS,
        fine_tune_epochs=max(2, POST_PRUNE_EPOCHS // N_ROUNDS),
        validation_data=(test_X, test_y),
        hard_masking=True,
        reconstruct_model=True,
        fit_verbose=0,
        fit_kwargs={"batch_size": BATCH_SIZE},
    )
    iterative_post = evaluate(result.model, test_X, test_y, "iterative_post")
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
    out_path = "sonar_mlp_pruning_results.csv"
    results.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"\nSaved all results to {out_path}")

    y_pred_final = (result.model.predict(test_X, verbose=0) > 0.5).astype(int).flatten()
    diff_path = f"sonar_confusion_diff_seed{seed}.png"
    confusion_diff_plot(
        test_y,
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
