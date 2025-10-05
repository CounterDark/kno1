import argparse
import tensorflow as tf
from pathlib import Path
import numpy as np


def getArgs():
    parser = argparse.ArgumentParser(
        description="Rozpoznawanie cyfr z obrazka przy użyciu TensorFlow"
    )
    parser.add_argument("-i", "--image", type=str, help="Ścieżka do pliku z obrazkiem")
    return parser.parse_args()


def getModel(x_train, y_train):
    if Path("model2.keras").exists():
        model = tf.keras.models.load_model("model2.keras")
    else:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(x_train, y_train, epochs=5)
    return model


def main():
    args = getArgs()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = getModel(x_train, y_train)
    # model.evaluate(x_test, y_test)

    print(args)

    if args.image:
        img = tf.keras.utils.load_img(
            args.image, target_size=(28, 28), color_mode="grayscale"
        )
        input_arr = tf.keras.utils.img_to_array(img) / 255.0
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = model.predict(input_arr)
        print(predictions.argmax())

    # model.summary()
    model.save("model2.keras")


if __name__ == "__main__":
    main()
