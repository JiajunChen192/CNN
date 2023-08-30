import numpy as np
import os
import pandas as pd
import tensorflow as tf
import librosa
from tensorflow.keras.optimizers import Adam

from utilities import *
from loss_and_training import *
from metadata_encoding import *
from modification_tensor_embedding import *
from convolution_layers import *
from decoder_network import *
from deconvolution_layers import *
from metadata_encoding import metadata_to_embedding


def encode_metadata(metadata_list):
    """
    Encode a list of metadata tuples into a tensor format suitable for the encoder based on the ASVspoof 2019 dataset.

    Args:
    - metadata_list: List of metadata tuples. Each tuple contains (SPEAKER_ID, AUDIO_FILE_NAME, SYSTEM_ID, KEY)

    """

    # Extract unique speaker IDs for one-hot encoding
    unique_speaker_ids = list(set([entry[0] for entry in metadata_list]))

    # Define the possible system IDs
    possible_system_ids = ['A{:02d}'.format(i) for i in range(1, 20)] + ['-']

    encoded_metadata = []

    for entry in metadata_list:
        speaker_id, _, system_id, key = entry

        # One-hot encode speaker ID
        speaker_encoding = [1 if speaker_id == unique_id else 0 for unique_id in unique_speaker_ids]

        # One-hot encode system ID
        system_encoding = [1 if system_id == possible_id else 0 for possible_id in possible_system_ids]

        # Encode the KEY
        key_encoding = [0] if key == 'bonafide' else [1]

        # Concatenate the encodings
        combined_encoding = speaker_encoding + system_encoding + key_encoding

        # Append to the encoded metadata list
        encoded_metadata.append(combined_encoding)

    # Convert to tensor
    encoded_metadata_tensor = tf.convert_to_tensor(encoded_metadata, dtype=tf.float32)

    return encoded_metadata_tensor


def load_metadata_from_protocol(protocol_file):
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
        metadata = []
        for line in lines:
            items = line.strip().split()
            speaker_id, audio_filename, system_id, _, key = items
            metadata.append((speaker_id, audio_filename, system_id, key))
        return metadata

# Paths based on the ASVspoof 2019 dataset directory structure
train_protocol_path = '/data2/jiajun/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
dev_protocol_path = '/data2/jiajun/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
eval_protocol_path = '/data2/jiajun/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

train_cover_audios_dir = '/data2/jiajun/LA/ASVspoof2019_LA_train/flac'
eval_cover_audios_dir = '/data2/jiajun/LA/ASVspoof2019_LA_eval/flac'

# Load metadata
train_metadatas = load_metadata_from_protocol(train_protocol_path)
print("First 10 entries of train_metadatas:")
print(train_metadatas[:10])

missing_files = []
for _, fname, _, _ in train_metadatas:
    file_path = os.path.join(train_cover_audios_dir, fname.split('.')[0] + '.flac')
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print(f"Found {len(missing_files)} missing files. Here are some of them:")
    print(missing_files[:10])
else:
    print("All files referenced in train_metadatas are present.")

dev_metadatas = load_metadata_from_protocol(dev_protocol_path)
eval_metadatas = load_metadata_from_protocol(eval_protocol_path)

# Load cover audio files using librosa
train_cover_audios_dataset = [librosa.load(os.path.join(train_cover_audios_dir, fname.split('.')[0] + '.flac'), sr=None)[0] for _, fname, _, _ in train_metadatas]
eval_cover_audios_dataset = [librosa.load(os.path.join(eval_cover_audios_dir, fname.split('.')[0] + '.flac'), sr=None)[0] for _, fname, _, _ in eval_metadatas]

# Initialize the encoder and decoder models
encoder = compile_encoder_with_metadata(encoder_network())
decoder = compile_decoder(decoder_network())

# Define the optimizer
optimizer = Adam(learning_rate=0.001)

# Training parameters
num_epochs = 10
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    problematic_files = []
    for _, fname, _, _ in train_metadatas:
        file_path = os.path.join(train_cover_audios_dir, fname.split('.')[0] + '.flac')
        try:
            librosa.load(file_path, sr=None)
        except Exception as e:
            problematic_files.append((file_path, str(e)))

    if problematic_files:
        print(f"Found {len(problematic_files)} problematic files when loading with librosa. Here are some of them:")
        print(problematic_files[:10])
    else:
        print("All files can be loaded correctly with librosa.")

    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Shuffle training data
    permutation = np.random.permutation(len(train_cover_audios_dataset))
    shuffled_train_cover_audios = [train_cover_audios_dataset[i] for i in permutation]
    shuffled_metadatas = [train_metadatas[i] for i in permutation]

    # Iterate over batches
    num_batches = len(train_cover_audios_dataset) // batch_size
    for batch in range(num_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        batch_train_cover_audios = shuffled_train_cover_audios[start:end]
        batch_train_metadatas = shuffled_metadatas[start:end]

        with tf.GradientTape() as tape:
            # Encode the batch_train_metadatas
            batch_train_metadatas_tensor = encode_metadata(batch_train_metadatas)

            batch_train_metadata_embeddings = [metadata_to_embedding(meta) for meta in batch_train_metadatas]
            modified_audios = encoder([batch_train_cover_audios, batch_train_metadata_embeddings])

            # Forward pass through the decoder
            reconstructed_metadatas = decoder(modified_audios)

            # Compute losses
            f_losses = [fidelity_loss(cover_audio, modified_audio) for cover_audio, modified_audio in
                        zip(batch_train_cover_audios, modified_audios)]
            a_losses = [accuracy_loss(metadata, reconstructed_metadata) for metadata, reconstructed_metadata in
                        zip(batch_train_metadatas, reconstructed_metadatas)]

            # Total loss for the batch
            total_loss = np.mean(f_losses) + np.mean(a_losses)

        # Backward pass: compute gradients and update weights
        grads = tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))

        print(f"Batch {batch + 1}/{num_batches} - Loss: {total_loss:.4f}")

# Evaluation loop on evaluation dataset
print("Evaluating on evaluation dataset")
eval_losses = []
num_eval_batches = len(eval_cover_audios_dataset) // batch_size
for batch in range(num_eval_batches):
    start = batch * batch_size
    end = (batch + 1) * batch_size
    batch_eval_cover_audios = eval_cover_audios_dataset[start:end]
    batch_eval_metadatas = eval_metadatas[start:end]

    # Forward pass through the encoder
    modified_audios = encoder([batch_eval_cover_audios, batch_eval_metadatas])

    # Forward pass through the decoder
    reconstructed_metadatas = decoder(modified_audios)

    # Compute losses
    f_losses = [fidelity_loss(cover_audio, modified_audio) for cover_audio, modified_audio in zip(batch_eval_cover_audios, modified_audios)]
    a_losses = [accuracy_loss(metadata, reconstructed_metadata) for metadata, reconstructed_metadata in zip(batch_eval_metadatas, reconstructed_metadatas)]

    # Total loss for the batch
    total_loss = np.mean(f_losses) + np.mean(a_losses)
    eval_losses.append(total_loss)

    print(f"Evaluation Batch {batch + 1}/{num_eval_batches} - Loss: {total_loss:.4f}")

print(f"Average Evaluation Loss: {np.mean(eval_losses):.4f}")

# Save the trained models
save_dir = '/data2/jiajun/train_data/'
encoder.save(os.path.join(save_dir, 'encoder_model.h5'))
decoder.save(os.path.join(save_dir, 'decoder_model.h5'))
