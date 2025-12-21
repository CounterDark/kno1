import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
from keras.models import Model

IMG_SIZE = 128
BATCH_SIZE = 32


def save(img, result="obrazek.png"):
    plt.imshow(img)
    plt.title(result)
    plt.gray()
    plt.savefig("saved/lab6/" + result)


def load_data():
    raw_dataset = keras.utils.image_dataset_from_directory(
        "data",  # katalog z obrazami
        labels=None,  # brak etykiet
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=True,
    )
    return raw_dataset.map(lambda x: x / 255.0)


class Autoencoder(Model):
    def __init__(self, latent_dim, encoder0=None, decoder0=None):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = (
            keras.Sequential(
                [
                    layers.Input(shape=(128, 128, 3)),
                    layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
                    layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
                    layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
                ]
            )
            if encoder0 is None
            else encoder0
        )
        self.decoder = (
            keras.Sequential(
                [
                    layers.Conv2DTranspose(
                        64, 3, strides=2, padding="same", activation="relu"
                    ),
                    layers.Conv2DTranspose(
                        128, 3, strides=2, padding="same", activation="relu"
                    ),
                    layers.Conv2DTranspose(
                        256, 3, strides=2, padding="same", activation="relu"
                    ),
                    layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
                ]
            )
            if decoder0 is None
            else decoder0
        )
        self.data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ]
        )

    def augment(self, dataset):
        augmented = dataset.map(lambda x: (self.data_augmentation(x), x))
        return augmented.prefetch(tf.data.AUTOTUNE)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


try:
    encoder = keras.models.load_model("saved/lab6/model_encoder.keras")
    decoder = keras.models.load_model("saved/lab6/model_decoder.keras")
    do_not_fit = True
except Exception as e:
    encoder = None
    decoder = None
    do_not_fit = False
    print(e)

autoencoder = Autoencoder(latent_dim=2, encoder0=encoder, decoder0=decoder)
autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
o_dataset = load_data()
dataset = o_dataset.concatenate(o_dataset)
dataset = autoencoder.augment(dataset)

if not do_not_fit:
    autoencoder.fit(
        dataset,
        epochs=10,
        shuffle=True,
    )

for batch in dataset.take(1):  # bierzemy pierwszy batch
    x_batch, y_batch = batch
    encoded_imgs = autoencoder.encoder(x_batch).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    decoded_imgs_to_compare = autoencoder(x_batch).numpy()

    for i in range(len(x_batch)):
        save(x_batch[i], result=f"x_test_{i}.png")
        save(decoded_imgs[i], result=f"decoded_imgs_{i}.png")
        save(decoded_imgs_to_compare[i], result=f"decoded_imgs_to_compare_{i}.png")
