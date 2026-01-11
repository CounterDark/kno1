#!/usr/bin/env python3
""" """
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.optimizers import Adam
from matplotlib.offsetbox import AnchoredText
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras_tuner as kt
import joblib
import json

RANDOM_SEED = 42

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class ModelBuilder:
    def __init__(self, model_type="dense", use_tuner=False, retrain=False):
        self.model_type = model_type
        self.use_tuner = use_tuner
        self.retrain = retrain
        self.model = None
        self.tuner = None
        self.summary = None
        self.batch_size = 50
        self.epochs = 5
        self.learning_rate = 0.001
        self.history = None
        self.loaded = False
        # Create saved dir
        self.path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "saved", "lab5"
        )
        print(f"Checking {self.path}")
        os.makedirs(
            self.path,
            exist_ok=True,
        )

    def load_or_build(self):
        model = None
        if (
            not self.retrain
            and Path(
                self.path
                + f"\\{self.model_type}{"_tuner" if self.use_tuner else ""}.keras"
            ).exists()
        ):
            model = keras.saving.load_model(
                self.path
                + f"\\{self.model_type}{"_tuner" if self.use_tuner else ""}.keras"
            )
            self.loaded = True
        if self.retrain or model is None:
            self.build_model()
        else:
            print("Loading model from saved")
            self.model = model

    def build_model(self):
        if self.model_type == "dense":
            if self.use_tuner:
                self.build_tuned_dense()
            else:
                self.build_dense()
        elif self.model_type == "conv":
            if self.use_tuner:
                self.build_tuned_convoluted()
            else:
                self.build_convoluted()
        else:
            raise NotImplementedError

    def build_dense_hp(self, hp):
        print("building dense hp model")
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(28, 28), batch_size=self.batch_size))
        model.add(keras.layers.Flatten())
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        model.add(keras.layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_convoluted_hp(self, hp):
        print("building convoluted hp model")
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(28, 28, 1), batch_size=self.batch_size))
        model.add(
            keras.layers.Conv2D(
                filters=hp.Int(
                    f"filters_convoluted_1", min_value=32, max_value=128, step=16
                ),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(
            keras.layers.Conv2D(
                filters=hp.Int(
                    f"filters_convoluted_2", min_value=64, max_value=256, step=32
                ),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_convoluted"),
                min_value=16,
                max_value=128,
                step=16,
                activation="relu",
            )
        )
        model.add(keras.layers.Dense(10, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_tuned_dense(self):
        print("Building tuned dense model...")
        tuner = kt.Hyperband(
            hypermodel=self.build_dense_hp,
            objective="val_accuracy",
            max_epochs=self.epochs,
            hyperband_iterations=2,
            overwrite=True,
            directory=self.path,
            project_name="dense_tuned",
        )
        self.tuner = tuner

    def build_tuned_convoluted(self):
        print("Building tuned convoluted model...")
        tuner = kt.Hyperband(
            hypermodel=self.build_convoluted_hp,
            objective="val_accuracy",
            max_epochs=self.epochs,
            hyperband_iterations=2,
            overwrite=True,
            directory=self.path,
            project_name="convoluted_tuned",
        )
        self.tuner = tuner

    def build_dense(self):
        print("Building dense model...")
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28), batch_size=self.batch_size),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model

    def build_convoluted(self):
        print("Building convoluted model...")
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28, 1), batch_size=self.batch_size),
                keras.layers.Conv2D(32, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model

    def train(self, X_train, X_test, y_train, y_test):
        # if self.loaded and not self.retrain:
        #     return
        print("Training...")
        if self.use_tuner:
            self.train_tuned(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )
        else:
            self.train_simple(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )

    def train_tuned(self, X_train, X_test, y_train, y_test):
        print("Training tuned dense model...")
        self.tuner.search(
            x=X_train,
            y=y_train,
            epochs=self.epochs,
            validation_data=(X_test, y_test),
        )
        best_params = self.tuner.get_best_hyperparameters(1)[0]
        if self.model_type == "dense":
            self.model = self.build_dense_hp(best_params)
        else:
            self.model = self.build_convoluted_hp(best_params)
        x_all = np.concatenate((X_train, X_test))
        y_all = np.concatenate((y_train, y_test))
        self.history = self.model.fit(
            x_all, y_all, epochs=self.epochs, batch_size=self.batch_size
        )

    def train_simple(self, X_train, X_test, y_train, y_test):
        self.history = self.model.fit(
            X_train, y_train, batch_size=self.batch_size, epochs=self.epochs
        )

    def plot_history_simple(self, loc="upper right"):
        """
        Save two images: {name}_accuracy_curve.png and {name}_loss_curve.png
        history_dict is expected to contain keys 'accuracy' and 'loss' (Keras history.history)
        """
        history_dict = self.history.history
        # accuracy
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(history_dict.get("accuracy", []), label="Accuracy (train)")
        if "val_accuracy" in history_dict:
            ax.plot(
                history_dict["val_accuracy"], linestyle="--", label="Accuracy (val)"
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        text = f"batch={self.batch_size}\nlr={self.learning_rate}\nepochs={self.epochs}"
        at = AnchoredText(text, loc=loc, prop={"family": "monospace"})
        at.patch.set_facecolor("white")
        at.patch.set_alpha(0.9)
        ax.add_artist(at)
        ax.legend()
        fig.savefig(
            self.path
            + f"/{self.model_type}_{"tuner" if self.use_tuner else ""}_accuracy_curve.png",
            dpi=150,
        )
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
        fig.savefig(
            self.path
            + f"/{self.model_type}_{"tuner" if self.use_tuner else ""}_loss_curve.png",
            dpi=150,
        )
        plt.close(fig)
        print(
            f"Saved curves: {self.model_type}_{"tuner" if self.use_tuner else ""}_accuracy_curve.png, {self.model_type}_{"tuner" if self.use_tuner else ""}_loss_curve.png"
        )

    def save_model(self):
        print("Saving model...")
        self.model.save(
            self.path + f"/{self.model_type}{"_tuner" if self.use_tuner else ""}.keras"
        )

    def predict(self, test_data):
        return self.model.predict(test_data)


def main(parsed_args):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Only grayscale
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(train_images.shape, test_images.shape)
    builder = ModelBuilder(
        model_type=parsed_args.model_type,
        use_tuner=parsed_args.use_tuner,
        retrain=parsed_args.retrain,
    )
    builder.load_or_build()
    print(f"Loaded: {builder.loaded} and retrain: {parsed_args.retrain}")
    if not builder.loaded or parsed_args.retrain:
        builder.train(
            X_train=train_images,
            X_test=test_images,
            y_train=train_labels,
            y_test=test_labels,
        )
        builder.plot_history_simple()
        builder.save_model()

    predict_data = test_images
    if parsed_args.predict and Path(parsed_args.path).exists():
        print("Loading predictions...")
        image = keras.utils.load_img(
            parsed_args.path, color_mode="grayscale", target_size=(28, 28)
        )
        input_arr = keras.utils.img_to_array(image)
        input_arr = 1 - input_arr
        input_arr = input_arr / 255.0
        predict_data = np.array([input_arr])

    predictions = builder.predict(predict_data)

    result = np.argmax(predictions[0])
    print(f"Wynik pedykcji: {CLASS_NAMES[result]} i wartość: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        dest="model_type",
        default="dense",
        choices=["dense", "conv"],
        help="Which model to build/train",
    )
    parser.add_argument(
        "--use-tuner",
        dest="use_tuner",
        action="store_true",
        help="If set, force retraining even if saved model exists",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="If set, force retraining even if saved model exists",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="If set, predict image from path",
    )
    parser.add_argument(
        "--path",
        default="",
    )
    args = parser.parse_args()
    main(args)
