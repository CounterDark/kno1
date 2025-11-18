import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import (
    callbacks,
    initializers,
    layers,
    models,
    optimizers,
    utils,
)

DATA_LOCAL = "public_resources/wine.csv"
NUM_FEATURES = 13


def load_csv(local_path=DATA_LOCAL):
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku {local_path}. Upewnij się, że istnieje."
        )
    print(f"Ładuję dane z pliku: {local_path}")
    return pd.read_csv(local_path, header=None)


def prepare_dataset(df):
    X = df.iloc[:, 1 : 1 + NUM_FEATURES].values.astype(np.float32)
    y = df.iloc[:, 0].values.astype(int)
    y = y - 1  # mapowanie klas 1,2,3 -> 0,1,2

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_oh = utils.to_categorical(y, num_classes=3)
    return X, y_oh, scaler


def build_model_a(input_shape, num_classes, seed=42):
    init = initializers.HeNormal(seed=seed)
    model = models.Sequential(name="Model_A_Shallow")
    model.add(layers.Input(shape=(input_shape,)))
    model.add(
        layers.Dense(
            32, activation="relu", kernel_initializer=init, name="dense_32_relu"
        )
    )
    model.add(
        layers.Dense(
            16, activation="relu", kernel_initializer=init, name="dense_16_relu"
        )
    )
    model.add(layers.Dense(num_classes, activation="softmax", name="output_softmax"))
    return model


def build_model_b(input_shape, num_classes, seed=24):
    model = models.Sequential(name="Model_B_Deep_Dropout")
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(64, activation="tanh", name="dense_64_tanh"))
    model.add(layers.Dropout(0.3, name="dropout_0.3"))
    model.add(layers.Dense(32, activation="tanh", name="dense_32_tanh"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output_softmax"))
    return model


def compile_and_train(
    model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, model_tag
):
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    log_dir = f"logs/{model_tag}"
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    chk = callbacks.ModelCheckpoint(
        f"best_{model_tag}.h5", save_best_only=True, monitor="val_accuracy"
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tb_cb, es, chk],
        verbose=2,
    )
    return history


def plot_history(histories, filename="training_curves.png"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for name, h in histories.items():
        plt.plot(h.history["accuracy"], label=f"{name} train")
        plt.plot(h.history["val_accuracy"], "--", label=f"{name} val")
        plt.title("Accuracy")
        plt.xlabel("epoch")
        plt.legend()

    plt.subplot(1, 2, 2)
    for name, h in histories.items():
        plt.plot(h.history["loss"], label=f"{name} train")
        plt.plot(h.history["val_loss"], "--", label=f"{name} val")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Zapisano wykresy do {filename}")


def predict_from_features(model_path, scaler, features):
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(arr_scaled)
    cls = np.argmax(probs, axis=1)[0]
    return cls + 1, probs[0]


def main(args):
    if args.predict:
        if not os.path.exists("best_model_used_for_prediction.h5"):
            raise FileNotFoundError(
                "Brak zapisanego modelu 'best_model_used_for_prediction.h5'. Uruchom trening."
            )
    if not os.path.exists("scaler.npy"):
        raise FileNotFoundError(
            "Brak zapisanego scaler'a 'scaler.npy'. Uruchom trening."
        )
    scaler = np.load("scaler.npy", allow_pickle=True).item()

    features = args.features
    if features is None or len(features) != NUM_FEATURES:
        raise ValueError(
            f"Podaj dokładnie {NUM_FEATURES} wartości cech przez --features"
        )
    cls, probs = predict_from_features(
        "best_model_used_for_prediction.h5", scaler, features
    )
    print(f"Przewidywana klasa wina: {cls}")
    print(f"Prawdopodobieństwa klas: {probs}")
    return

    # Trening
    df = load_csv()
    X, y_oh, scaler = prepare_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_oh, test_size=0.2, random_state=42, stratify=y_oh
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
    )

    print(
        f"Rozmiary: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    model_a = build_model_a(NUM_FEATURES, num_classes=3)
    model_b = build_model_b(NUM_FEATURES, num_classes=3)

    model_a.summary()
    model_b.summary()

    histories = {}
    hist_a = compile_and_train(
        model_a,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_tag="model_a",
    )
    histories["Model_A"] = hist_a

    hist_b = compile_and_train(
        model_b,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_tag="model_b",
    )
    histories["Model_B"] = hist_b

    best_a = (
        tf.keras.models.load_model("best_model_a.h5")
        if os.path.exists("best_model_a.h5")
        else model_a
    )
    best_b = (
        tf.keras.models.load_model("best_model_b.h5")
        if os.path.exists("best_model_b.h5")
        else model_b
    )

    loss_a, acc_a = best_a.evaluate(X_test, y_test, verbose=0)
    loss_b, acc_b = best_b.evaluate(X_test, y_test, verbose=0)

    print(
        f"Test results:\n Model_A - loss={loss_a:.4f}, acc={acc_a:.4f}\n Model_B - loss={loss_b:.4f}, acc={acc_b:.4f}"
    )

    if acc_a >= acc_b:
        chosen_model = best_a
        chosen_name = "Model_A"
        chosen_acc = acc_a
    else:
        chosen_model = best_b
        chosen_name = "Model_B"
        chosen_acc = acc_b

    chosen_model.save("best_model_used_for_prediction.h5")
    np.save("scaler.npy", scaler)

    plot_history(histories, filename="training_curves.png")

    with open("train_report.txt", "w") as f:
        f.write(f"Model wybrany: {chosen_name}\n")
    f.write(f"Test accuracy A: {acc_a:.4f}\n")
    f.write(f"Test accuracy B: {acc_b:.4f}\n")
    f.write(f"Wybrany accuracy: {chosen_acc:.4f}\n")

    print(
        "Trening zakończony. Zapisano: best_model_used_for_prediction.h5, scaler.npy, training_curves.png, train_report.txt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trening i predykcja modeli klasyfikacji win (UCI)."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Uruchom trening (domyślnie jeśli brak --predict)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Wykonaj predykcję na podstawie podanych cech",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=float,
        help=f"Lista {NUM_FEATURES} cech dla predykcji",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if not args.predict:
        args.train = True

    main(args)
