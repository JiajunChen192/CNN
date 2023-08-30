from tensorflow.keras.layers import Input, Conv1D, ReLU, BatchNormalization, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

def decoder_network(input_shape=(None, 1)):
    steganographic_audio_input = Input(shape=input_shape, name="steganographic_audio")

    x = Conv1D(128, 3, padding='same', name="Decoder_Conv1")(steganographic_audio_input)
    x = ReLU(name="Decoder_ReLU1")(x)
    x = BatchNormalization(name="Decoder_BN1")(x)
    x = Conv1D(64, 3, padding='same', name="Decoder_Conv2")(x)
    x = ReLU(name="Decoder_ReLU2")(x)
    x = BatchNormalization(name="Decoder_BN2")(x)
    x = GlobalAveragePooling1D()(x)
    decoded_metadata = Dense(1024, activation='sigmoid', name="Decoded_Metadata")(x)

    return Model(inputs=steganographic_audio_input, outputs=decoded_metadata, name="Decoder")

if __name__ == "__main__":
    decoder_model = decoder_network()
    decoder_model.summary()

