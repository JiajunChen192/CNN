import tensorflow as tf
from keras import Input
from tensorflow.keras.layers import Conv1D, ReLU, BatchNormalization
from tensorflow.keras.models import Model


def encoder_network(input_shape=(None, 1)):
    cover_audio_input = tf.keras.layers.Input(shape=input_shape, name="cover_audio")

    x = Conv1D(64, 3, padding='same', name="Conv1")(cover_audio_input)
    x = ReLU(name="ReLU1")(x)
    x = BatchNormalization(name="BN1")(x)

    encoded_cover = Conv1D(1024, 3, padding='same', name="Conv2")(x)
    encoded_cover = ReLU(name="ReLU2")(encoded_cover)
    encoded_cover = BatchNormalization(name="BN2")(encoded_cover)

    return Model(inputs=cover_audio_input, outputs=encoded_cover, name="Encoder")

def audio_steganography_model(N):
    # 1. Input Layer: Accepts the cover audio tensor.
    # Assuming the audio tensor is of shape (num_samples, 1), where num_samples is the length of the audio.
    input_audio = Input(shape=(None, 1), name="cover_audio")

    # 2. Convolutional Layers:

    # Conv1: Extracts low-level features from the cover audio.
    x = Conv1D(filters=N, kernel_size=1000, padding="same", name="Conv1")(input_audio)
    x = ReLU(name="ReLU1")(x)
    x = BatchNormalization(name="BN1")(x)

    # Conv2: Extracts mid-level features.
    x = Conv1D(filters=2 * N, kernel_size=5, padding="same", name="Conv2")(x)
    x = ReLU(name="ReLU2")(x)
    x = BatchNormalization(name="BN2")(x)

    model = tf.keras.Model(inputs=input_audio, outputs=x)
    return model


# For demonstration purposes, let's assume N = 64
model = audio_steganography_model(N=64)
model.summary()
