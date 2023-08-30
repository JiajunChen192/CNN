from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU
from deconvolution_layers import complete_audio_steganography_model
from tensorflow.keras.optimizers import Adam


def compile_encoder_with_metadata(encoder):
        cover_audio = tf.keras.layers.Input(shape=(None, 1), name="cover_audio")
        metadata = tf.keras.layers.Input(shape=(1024,), name="metadata")  # Updated shape

        x = tf.keras.layers.Conv1D(64, 3, padding='same', name="Conv1")(cover_audio)
        x = tf.keras.layers.ReLU(name="ReLU1")(x)
        x = tf.keras.layers.BatchNormalization(name="BN1")(x)

        x = tf.keras.layers.Conv1D(128, 3, padding='same', name="Conv2")(x)
        x = tf.keras.layers.ReLU(name="ReLU2")(x)
        x = tf.keras.layers.BatchNormalization(name="BN2")(x)

        # Adjusting the Dense layer to accommodate the change
        metadata_embedding = tf.keras.layers.Dense(128, activation='relu', name="Metadata_Dense1")(metadata)
        metadata_embedding = tf.keras.layers.Reshape((1, 128), name="Reshape_Metadata")(metadata_embedding)

        x = tf.keras.layers.Add(name="Combined_Features")([x, metadata_embedding])

        x = tf.keras.layers.Conv1DTranspose(64, 3, padding='same', name="DeConv1")(x)
        x = tf.keras.layers.ReLU(name="ReLU_DeConv1")(x)
        x = tf.keras.layers.BatchNormalization(name="BN_DeConv1")(x)

        x = tf.keras.layers.Conv1DTranspose(1, 3, padding='same', name="DeConv2")(x)
        x = tf.keras.layers.ReLU(name="ReLU_DeConv2")(x)
        x = tf.keras.layers.BatchNormalization(name="BN_DeConv2")(x)

        modification_tensor = tf.keras.layers.Activation('tanh', name="Modification_Tensor")(x)

        encoder_with_metadata = tf.keras.Model(inputs=[cover_audio, metadata], outputs=modification_tensor,
                                               name="Encoder")

        return encoder_with_metadata



def compile_decoder(model):
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
    return model


def fidelity_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def accuracy_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)


def decoder_network(N, metadata_dim):
    # Input for steganographic audio (which is modified cover audio)
    input_audio = Input(shape=(None, 1), name="steganographic_audio")

    # Mirrored version of the encoder
    # Deconv Layers
    x = Conv1D(filters=2 * N, kernel_size=3, padding="same", name="Decoder_Conv1")(input_audio)
    x = ReLU(name="Decoder_ReLU1")(x)
    x = BatchNormalization(name="Decoder_BN1")(x)

    x = Conv1D(filters=N, kernel_size=3, padding="same", name="Decoder_Conv2")(x)
    x = ReLU(name="Decoder_ReLU2")(x)
    x = BatchNormalization(name="Decoder_BN2")(x)

    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Metadata Decoding Layer
    decoded_metadata = Dense(units=metadata_dim, activation='sigmoid', name="Decoded_Metadata")(x)

    model = tf.keras.Model(inputs=input_audio, outputs=decoded_metadata)
    return model


# Loss & Training
def compile_model(model, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')  # Using MSE for fidelity loss for now
    return model


# Create and compile the models
encoder_model = complete_audio_steganography_model(N=64, metadata_dim=128)
decoder_model = decoder_network(N=64, metadata_dim=128)
compiled_encoder = compile_model(encoder_model)
compiled_decoder = compile_model(decoder_model)

compiled_encoder.summary()
compiled_decoder.summary()
