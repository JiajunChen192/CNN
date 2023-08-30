import numpy as np
import tensorflow as tf
import base64
# Importing the necessary modules
from utilities import prepare_data, save_audio
from loss_and_training import compile_encoder_with_metadata, compile_decoder, fidelity_loss, accuracy_loss
from convolution_layers import encoder_network
from decoder_network import decoder_network
import librosa.display
import matplotlib.pyplot as plt
import librosa



# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Step 1: Reading the Encrypted Message and Encoding to Fixed Length Binary
def read_and_encode_message(file_path):
    with open(file_path, 'r') as file:
        message = file.read()
    # Convert the message to Base64
    encoded_message = base64.b64encode(message.encode()).decode()
    # Convert the Base64 encoded message to binary
    binary_message = ''.join(format(ord(i), '08b') for i in encoded_message)
    # Pad the binary message to 1024 bits
    while len(binary_message) < 1024:
        binary_message += '0'
    return np.array([int(bit) for bit in binary_message])

# Step 2: Test the Pipeline
from modification_tensor_embedding import modify_cover_audio


def steganography_pipeline(cover_audio, metadata):
    # Load and compile the encoder and decoder models
    encoder = encoder_network()
    decoder = decoder_network()

    encoder_with_metadata = compile_encoder_with_metadata(encoder)
    compiled_decoder = compile_decoder(decoder)

    # Use the encoder to generate the modification tensor
    modification_tensor = encoder_with_metadata([cover_audio, np.expand_dims(metadata, axis=0)])
    modification_tensor = tf.squeeze(modification_tensor, axis=-1)

    print("Shape of cover_audio:", cover_audio.shape)
    print("First 10 values of cover_audio:", cover_audio[:10])
    print("Shape of modification_tensor:", modification_tensor.shape)
    print("First 10 values of modification_tensor:", modification_tensor[:10])
    # Find the index of the first non-zero value in the modification_tensor
    first_non_zero_idx = next((i for i, value in enumerate(modification_tensor) if value != 0), None)
    # Print the number of leading zeros
    print(f"Number of leading zeros in modification_tensor: {first_non_zero_idx}")

    # Use modify_cover_audio to combine the modification tensor with the original cover audio
    alpha = 0.01  # Embedding strength; you can adjust this value
    modified_audio = modify_cover_audio(cover_audio, modification_tensor, alpha)

    # Print shape of modified_audio
    print(f"Shape of modified_audio: {modified_audio.shape}")

    # Use the decoder to extract the metadata from the steganographic audio
    reconstructed_metadata_full = compiled_decoder(modified_audio)
    reconstructed_metadata = tf.reduce_mean(reconstructed_metadata_full, axis=0)

    # Print shape of reconstructed_metadata
    print(f"Shape of reconstructed_metadata: {reconstructed_metadata.shape}")

    # Calculate the fidelity loss and accuracy loss
    f_loss = fidelity_loss(cover_audio, modified_audio)
    a_loss = accuracy_loss(metadata, reconstructed_metadata)

    return modified_audio, reconstructed_metadata, f_loss, a_loss


def visualize_audio_files_separately(input_path, output_path):
    # Load the audio files
    input_audio, _ = librosa.load(input_path, sr=16000)
    output_audio, _ = librosa.load(output_path, sr=16000)

    # Plot input audio
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(input_audio, sr=16000)
    plt.title('Waveform of input.wav')
    plt.tight_layout()

    # Plot output audio
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(output_audio, sr=16000)
    plt.title('Waveform of output.wav')
    plt.tight_layout()

    plt.savefig('/home/jiajunchen/Embedding Generation using CNN for Metadata Audio Steganography/waveform_comparison.png')
    plt.show()

# Load Data
cover_audio = prepare_data('input.wav')
original_metadata = read_and_encode_message('encrypted_message.txt')

# Run the pipeline
modified_audio, reconstructed_metadata, f_loss, a_loss = steganography_pipeline(cover_audio, original_metadata)
save_audio(modified_audio.numpy().squeeze(), 'output.wav')

print(f"Shape of modified_audio: {modified_audio.shape}")
visualize_audio_files_separately('input.wav', 'output.wav')

# Load the input.wav file
desired_sampling_rate = 16000  # example value; adjust as needed
y_input, sr_input= librosa.load('input.wav', sr=desired_sampling_rate)

# Load the output.wav file
y_output, sr_output = librosa.load('output.wav', sr=16000)

# Output Results
print("Modified Audio:", modified_audio)
print("Reconstructed Metadata:", reconstructed_metadata)
print("Fidelity Loss:", f_loss)
print("Accuracy Loss:", a_loss)
print(f"Sampling rate of input.wav: {sr_input} Hz")
print(f"Sampling rate of output.wav: {sr_output} Hz")
