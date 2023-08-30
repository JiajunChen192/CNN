from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv1DTranspose, Activation
from tensorflow.keras.layers  import Dense
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU

def complete_audio_steganography_model(N, metadata_dim):
    # 1. Input Layer for Audio
    input_audio = Input(shape=(None, 1), name="cover_audio")

    # 2. Convolutional Layers:
    # Conv1
    x = Conv1D(filters=N, kernel_size=3, padding="same", name="Conv1")(input_audio)
    x = ReLU(name="ReLU1")(x)
    x = BatchNormalization(name="BN1")(x)

    # Conv2
    x = Conv1D(filters=2 * N, kernel_size=3, padding="same", name="Conv2")(x)
    x = ReLU(name="ReLU2")(x)
    audio_features = BatchNormalization(name="BN2")(x)

    # 3. Metadata Encoding Layer:
    # Input Layer for Metadata
    input_metadata = Input(shape=(metadata_dim,), name="metadata")
    metadata_features = Dense(units=2 * N, activation='relu', name="Metadata_Dense1")(input_metadata)
    metadata_features = tf.keras.layers.Reshape(target_shape=(-1, 2 * N), name="Reshape_Metadata")(metadata_features)
    combined_features = Add(name="Combined_Features")([audio_features, metadata_features])

    # 4. Deconvolutional Layers:
    # DeConv1
    x = Conv1DTranspose(filters=N, kernel_size=1000, padding="same", name="DeConv1")(combined_features)
    x = ReLU(name="ReLU_DeConv1")(x)
    x = BatchNormalization(name="BN_DeConv1")(x)

    # DeConv2
    x = Conv1DTranspose(filters=1, kernel_size=5, padding="same", name="DeConv2")(x)
    x = ReLU(name="ReLU_DeConv2")(x)
    deconv_features = BatchNormalization(name="BN_DeConv2")(x)

    # 5. Modification Tensor Generation:
    modification_tensor = Activation("sigmoid", name="Modification_Tensor")(deconv_features)

    model = tf.keras.Model(inputs=[input_audio, input_metadata], outputs=modification_tensor)
    return model

# Assuming N = 64 and metadata_dim = 128 (for demonstration purposes)
complete_model = complete_audio_steganography_model(N=64, metadata_dim=128)
complete_model.summary()
