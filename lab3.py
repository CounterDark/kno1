#!/usr/bin/env python3
"""
wine_methods_only.py
Function-only reimplementation (no classes) that:
 - loads and shuffles the wine csv from public_resources/wine.csv
 - one-hot encodes classes
 - scales features
 - defines two model-builder functions
 - trains or loads models (saved as <name>_model.keras)
 - plots training curves
 - performs a one-sample prediction from argparse inputs

Usage (train & predict example):
  python wine_methods_only.py --recalc --model second --epochs 15 --batch_size 32 --lr 0.001
  python wine_methods_only.py --predict --alcohol 13.2 --malic-acid 1.78 ... --proline 1050
"""
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import keras_tuner as kt
import joblib
import json

RANDOM_SEED = 42


# -----------------------
# Data functions
# -----------------------
def load_and_shuffle_csv(path="public_resources/wine.csv", random_state=RANDOM_SEED):
    """Load CSV, assume first column 'class' or headerless with class first.
    Returns pandas DataFrame with 'class' column if detected, and rest numeric columns.
    """
    df = pd.read_csv(path)
    # If headerless (original UCI), the file may have no header. Try to detect:
    if df.columns[0] != "class" and df.shape[1] == 14:
        # assume first col is class; rename for clarity
        cols = ["class"] + [f"f{i}" for i in range(1, 14)]
        df.columns = cols
    if "class" not in df.columns:
        raise ValueError(
            "CSV must contain class in first column or header named 'class'."
        )
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def make_one_hot_and_split(df, test_size=0.2, random_state=RANDOM_SEED):
    """Take dataframe with 'class' column, return X_train, X_test, y_train, y_test, scaler placeholder."""
    y = pd.get_dummies(df["class"]).astype(int)
    X = df.drop(columns=["class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_scaler_and_transform(X_train):
    """Fit StandardScaler on X_train and return (scaler, X_train_transformed)"""
    scaler_exists = Path(f"saved/lab3/scaler.pk1").exists()

    scaler = (
        StandardScaler() if not scaler_exists else joblib.load(f"saved/lab3/scaler.pk1")
    )
    X_train_t = scaler.fit_transform(X_train)
    if not scaler_exists:
        joblib.dump(scaler, f"saved/lab3/scaler.pk1")
    return scaler, X_train_t


def transform_with_scaler(scaler, X):
    return scaler.transform(X)


def normalize_data(X_train):
    train_normalizer = layers.Normalization(axis=1)
    train_normalizer.adapt(np.array(X_train))
    return train_normalizer


# -----------------------
# Model builder functions
# -----------------------
def build_model_simple(normalizer, input_dim=13, learning_rate=0.001):
    """Small model similar to 'first'"""
    model = Sequential(
        [
            layers.Input(shape=(input_dim,), name="input"),
            normalizer,
            layers.Dense(32, activation="relu", name="hidden_1"),
            layers.Dense(3, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_deep(normalizer, input_dim=13, learning_rate=0.001):
    """Deeper model similar to 'second' but with slightly different layout (still many layers)."""
    model = Sequential(
        [
            layers.Input(shape=(input_dim,), name="input"),
            normalizer,
            layers.Dense(
                256, activation="relu", kernel_initializer="HeNormal", name="h1"
            ),
            layers.Dense(
                128, activation="relu", kernel_initializer="HeNormal", name="h2"
            ),
            layers.Dense(
                64, activation="relu", kernel_initializer="HeNormal", name="h3"
            ),
            layers.Dense(
                32, activation="relu", kernel_initializer="HeNormal", name="h4"
            ),
            layers.Dense(
                16, activation="relu", kernel_initializer="HeNormal", name="h5"
            ),
            layers.Dropout(0.2, name="dropout"),
            layers.Dense(3, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_hyper(
    X_train, y_train, X_test, y_test, normalizer, input_dim=13, learning_rate=0.001
):

    def hyper_build(hp: kt.HyperParameters):
        model = Sequential()
        model.add(layers.Input(shape=(input_dim,), name="input"))
        model.add(normalizer)
        model.add(
            layers.Dense(
                hp.Int("units_1", min_value=16, max_value=256, step=16),
                activation="relu",
                name="hidden_1",
            )
        )
        if hp.Boolean("use_dropout"):
            model.add(layers.Dropout(0.2, name="dropout"))
        model.add(
            layers.Dense(
                3,
                activation="softmax",
                name="output",
            )
        )
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Float(
                    "lr", min_value=0.00001, max_value=0.1, sampling="log"
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.BayesianOptimization(
        hypermodel=hyper_build,
        objective="val_accuracy",
        max_trials=10,
        seed=RANDOM_SEED,
        directory="saved",
    )

    tuner.search_space_summary()

    tuner.search(X_train, y_train, epochs=4, validation_data=(X_test, y_test))

    best_model = tuner.get_best_models(num_models=1)[0]
    best_params: kt.HyperParameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model.summary()

    with open(f"saved/lab3/best_hparams.json", "w") as f:
        json.dump(best_params.values, f, indent=2)

    return best_model


# -----------------------
# Training / saving / loading functions
# -----------------------
def train_and_save_model(
    name,
    model_builder_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    normalizer,
    recalc=True,
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
):
    """
    If model file exists and recalc is False -> load and return model.
    Otherwise train new model and save it as <name>_model.keras
    """
    model_path = Path(f"./saved/lab3/{name}_model.keras")
    if model_path.exists() and not recalc:
        print(f"Loading existing model from {model_path}")
        return tf.keras.models.load_model(model_path)
    print(
        f"Training model '{name}' (epochs={epochs}, batch_size={batch_size}, lr={learning_rate})..."
    )
    model = model_builder_fn(
        X_train,
        y_train,
        X_test,
        y_test,
        normalizer=normalizer,
        learning_rate=learning_rate,
    )
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2
    )
    plot_history_simple(
        history.history,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        name=name,
    )
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model


# -----------------------
# Plotting
# -----------------------
def plot_history_simple(
    history_dict, batch_size, epochs, learning_rate, name, loc="upper right"
):
    """
    Save two images: {name}_accuracy_curve.png and {name}_loss_curve.png
    history_dict is expected to contain keys 'accuracy' and 'loss' (Keras history.history)
    """
    # accuracy
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(history_dict.get("accuracy", []), label="Accuracy (train)")
    if "val_accuracy" in history_dict:
        ax.plot(history_dict["val_accuracy"], linestyle="--", label="Accuracy (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    text = f"batch={batch_size}\nlr={learning_rate}\nepochs={epochs}"
    at = AnchoredText(text, loc=loc, prop={"family": "monospace"})
    at.patch.set_facecolor("white")
    at.patch.set_alpha(0.9)
    ax.add_artist(at)
    ax.legend()
    fig.savefig(f"saved/lab3/{name}_accuracy_curve.png", dpi=150)
    plt.close(fig)

    # loss
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(history_dict.get("loss", []), label="Loss (train)")
    if "val_loss" in history_dict:
        ax.plot(history_dict["val_loss"], linestyle="--", label="Loss (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, max(1.0, max(history_dict.get("loss", [1.0])) * 1.1))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    at = AnchoredText(text, loc=loc, prop={"family": "monospace"})
    at.patch.set_facecolor("white")
    at.patch.set_alpha(0.9)
    ax.add_artist(at)
    ax.legend()
    fig.savefig(f"saved/lab3/{name}_loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved curves: {name}_accuracy_curve.png, {name}_loss_curve.png")


# -----------------------
# Prediction helpers
# -----------------------
def args_to_feature_vector(args):
    """Collect 13 feature args in expected order and return list"""
    order = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od_diluted_wines",
        "proline",
    ]
    # If malic_acid unused, we include it because the provided classmate had it; keep order consistent.
    vec = []
    for key in order:
        val = getattr(args, key, None)
        if val is None:
            raise ValueError(f"Missing argument: --{key.replace('_','-')}")
        vec.append(float(val))
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def predict_one_sample(model, feature_vector):
    probs = model.predict(feature_vector, verbose=0)[0]
    pred = int(np.argmax(probs)) + 1
    return pred, probs


# -----------------------
# Main orchestration
# -----------------------
def main(parsed_args):
    # 1) load, prepare, split
    df = load_and_shuffle_csv(path="public_resources/wine.csv")
    X_train, X_test, y_train, y_test = make_one_hot_and_split(df, test_size=0.2)
    # scaler, X_train = fit_scaler_and_transform(X_train_raw)
    normalizer = normalize_data(X_train)

    # 2) select model builder
    model_name = parsed_args.model
    if model_name == "first":
        builder = build_model_simple
    elif model_name == "second":
        builder = build_model_deep
    elif model_name == "hyper":
        builder = build_model_hyper
    else:
        raise ValueError("Unknown model name; choose 'first' or 'second'")

    # 3) train or load
    model = train_and_save_model(
        name=model_name,
        model_builder_fn=builder,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        recalc=parsed_args.recalc,
        batch_size=parsed_args.batch_size,
        epochs=parsed_args.epochs,
        learning_rate=parsed_args.lr,
        normalizer=normalizer,
    )

    # 4) if predict flag provided -> run single-sample prediction using CLI-provided feature flags
    if parsed_args.predict:
        feat_vec = args_to_feature_vector(parsed_args)
        pred_class, probs = predict_one_sample(model, feat_vec)
        print("Predicted class:", pred_class)
        print("Class probabilities:", probs)

    # 5) report quick test-set accuracy (optional)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test set -> loss: {loss:.4f}, acc: {acc:.4f}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="second",
        choices=["first", "second", "hyper"],
        help="Which model to build/train/load",
    )
    parser.add_argument(
        "--recalc",
        action="store_true",
        help="If set, force retraining even if saved model exists",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run one-sample prediction using provided features",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    # 13 feature args (keeps similar names to classmate)
    parser.add_argument("--malic-acid", dest="malic_acid", type=float, default=0.0)
    parser.add_argument("--alcohol", type=float, default=0.0)
    parser.add_argument("--ash", type=float, default=0.0)
    parser.add_argument(
        "--alcalinity-of-ash", dest="alcalinity_of_ash", type=float, default=0.0
    )
    parser.add_argument("--magnesium", type=float, default=0.0)
    parser.add_argument(
        "--total-phenols", dest="total_phenols", type=float, default=0.0
    )
    parser.add_argument("--flavanoids", type=float, default=0.0)
    parser.add_argument(
        "--nonflavanoid-phenols", dest="nonflavanoid_phenols", type=float, default=0.0
    )
    parser.add_argument("--proanthocyanins", type=float, default=0.0)
    parser.add_argument(
        "--color-intensity", dest="color_intensity", type=float, default=0.0
    )
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument(
        "--od-diluted-wines", dest="od_diluted_wines", type=float, default=0.0
    )
    parser.add_argument("--proline", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
