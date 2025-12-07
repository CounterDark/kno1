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


class ModelBuilder:
    def __init__(self, model_type="dense", use_tuner=False, retrain=False):
        self.model_type = model_type
        self.use_tuner = use_tuner
        self.retrain = retrain
        self.model = None
        self.summary = None
        self.batch_size = 50
        self.epochs = 5
        self.learning_rate = 0.001
        self.history = None
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
        model = keras.saving.load_model(
            self.path + f"/{self.model_type}{"_tuner" if self.use_tuner else ""}.keras"
        )
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

    def build_convoluted_hp(self, hp):
        print("building convoluted hp model")

    def build_tuned_dense(self):
        print("Building tuned dense model...")

    def build_tuned_convoluted(self):
        print("Building tuned convoluted model...")

    def build_dense(self):
        print("Building dense model...")
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28), batch_size=self.batch_size),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax),
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
    builder.train(
        X_train=train_images,
        X_test=test_images,
        y_train=train_labels,
        y_test=test_labels,
    )
    builder.plot_history_simple()
    builder.save_model()


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
    args = parser.parse_args()
    main(args)
