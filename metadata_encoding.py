import tensorflow as tf
from keras import Input
from tensorflow.keras.layers  import ReLU, Conv1D, BatchNormalization
from tensorflow.keras.layers import Add, Embedding, Input, Flatten, Concatenate
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Model

def metadata_encoder_network(input_shape=(1024,)):
    metadata_input = Input(shape=input_shape, name="metadata")
    dense1 = Dense(1024, activation='relu', name="Metadata_Dense1")(metadata_input)
    reshaped_metadata = Reshape((1, 1024), name="Reshape_Metadata")(dense1)
    return Model(inputs=metadata_input, outputs=reshaped_metadata, name="Metadata_Encoder")

def create_one_hot_embedding(unique_values):
    return {value: np.eye(len(unique_values))[i] for i, value in enumerate(unique_values)}

def generate_one_hot_mappings(train_metadatas):
    unique_speaker_ids = sorted(list(set([meta[0] for meta in train_metadatas])))
    unique_file_ids = sorted(list(set([meta[1] for meta in train_metadatas])))
    unique_system_ids = sorted(list(set([meta[2] for meta in train_metadatas])))
    unique_labels = sorted(list(set([meta[3] for meta in train_metadatas])))

    # Create one-hot embeddings for each unique value
    speaker_id_to_embedding = create_one_hot_embedding(unique_speaker_ids)
    file_id_to_embedding = create_one_hot_embedding(unique_file_ids)
    system_id_to_embedding = create_one_hot_embedding(unique_system_ids)
    label_to_embedding = create_one_hot_embedding(unique_labels)

    return speaker_id_to_embedding, file_id_to_embedding, system_id_to_embedding, label_to_embedding


def metadata_to_embedding(metadata):
    speaker_id, file_id, system_id, label = metadata

    # Convert speaker_id to embedding
    speaker_id_embedding = speaker_id_to_embedding[speaker_id]

    # Convert file_id to embedding
    file_id_embedding = file_id_to_embedding[file_id]

    # Convert system_id to embedding
    system_id_embedding = system_id_to_embedding[system_id]

    # Convert label to embedding
    label_embedding = label_to_embedding[label]

    # Combine embeddings
    combined_embedding = np.concatenate([speaker_id_embedding, file_id_embedding, system_id_embedding, label_embedding],
                                        axis=-1)

    return combined_embedding


def audio_steganography_model_with_metadata(N, metadata_dim):
    # 1. Input Layer for Audio
    input_audio = Input(shape=(None, 1), name="cover_audio")

    # 2. Convolutional Layers:
    x = Conv1D(filters=N, kernel_size=3, padding="same", name="Conv1")(input_audio)
    x = ReLU(name="ReLU1")(x)
    x = BatchNormalization(name="BN1")(x)
    x = Conv1D(filters=2 * N, kernel_size=3, padding="same", name="Conv2")(x)
    x = ReLU(name="ReLU2")(x)
    audio_features = BatchNormalization(name="BN2")(x)

    # 3. Metadata Encoding Layer:
    input_metadata = Input(shape=(metadata_dim,), name="metadata")
    metadata_features = Dense(units=2 * N, activation='relu', name="Metadata_Dense1")(input_metadata)
    metadata_features = tf.keras.layers.Reshape(target_shape=(-1, 2 * N), name="Reshape_Metadata")(metadata_features)
    combined_features = Add(name="Combined_Features")([audio_features, metadata_features])

    model = tf.keras.Model(inputs=[input_audio, input_metadata], outputs=combined_features)
    return model


def metadata_encoder(input_metadata):
    speaker_id_input = Input(shape=(1,), name="speaker_id_input")
    system_id_input = Input(shape=(1,), name="system_id_input")

    # These numbers should be replaced with actual counts of unique speaker and system IDs.
    num_unique_speaker_ids = 10  # Placeholder value
    num_unique_system_ids = 5  # Placeholder value

    speaker_embedding = Embedding(output_dim=8, input_dim=num_unique_speaker_ids, input_length=1,
                                  name="speaker_embedding")(speaker_id_input)
    system_embedding = Embedding(output_dim=4, input_dim=num_unique_system_ids, input_length=1,
                                 name="system_embedding")(system_id_input)

    # Flatten the embeddings
    speaker_embedding = Flatten()(speaker_embedding)
    system_embedding = Flatten()(system_embedding)

    concatenated = Concatenate()([speaker_embedding, system_embedding])

    model = Model(inputs=[speaker_id_input, system_id_input], outputs=concatenated)

    encoded_metadata = model.predict([input_metadata['speaker_id'], input_metadata['system_id']])

    return encoded_metadata

# Assuming N = 64 and metadata_dim = 128 (for demonstration purposes)
model_with_metadata = audio_steganography_model_with_metadata(N=64, metadata_dim=128)
model_with_metadata.summary()
