import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam

from src.pruning_flow import run_pruning_seeded_benchmark

# --- Synthetic demo data ---
X = np.random.uniform(-1, 1, (1000, 2))
y = (X[:, 0] ** 2 + X[:, 1] > 0.5).astype(int)

n_in = X.shape[1]


def build_model():
    m = Sequential(
        [
            Dense(64, activation="relu", input_shape=(n_in,)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
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


summary_df, per_seed_df = run_pruning_seeded_benchmark(
    build_model,
    X,
    y,
    n_seeds=10,
    test_size=0.2,
    pre_prune_epochs=65,
    post_prune_epochs=35,
    prune_ratio=0.2,
    hard_masking=False,  # set to True to keep pruned edges at 0 during fine-tune
    fit_verbose=0,
)

print("########################")
print("## Summary DataFrame: ##")
print("########################")
print(summary_df)

print("")

print("#########################")
print("## Per-Seed DataFrame: ##")
print("#########################")
print(per_seed_df)
