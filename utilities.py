import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf


def binary_string_to_array(binary_str):
    """Converts a binary string to a numpy array."""
    return np.array([int(bit) for bit in binary_str], dtype=np.float32)


def array_to_binary_string(binary_array):
    """Converts a numpy array to a binary string."""
    return ''.join([str(int(bit)) for bit in binary_array])


def load_audio_and_message(audio_path, message_path):

    # Load audio
    sample_rate, audio_data = wavfile.read(audio_path)
    audio_data = audio_data.astype(np.float32) / 32767.0  # Convert to float32 in range [-1, 1]
    audio_data = np.expand_dims(audio_data, axis=-1)  # Add channel dimension

    # Load message
    with open(message_path, 'r') as file:
        original_metadata_str = file.read().strip()
    original_metadata = binary_string_to_array(original_metadata_str)

    return audio_data, original_metadata
def prepare_data(audio_path):
    # Load audio data using librosa
    audio_data, _ = librosa.load(audio_path, sr=None, mono=True)
    audio_data = np.expand_dims(audio_data, axis=-1)  # Add channel dimension
    return audio_data

def save_audio(audio_data, path):
    # Save the audio data to the given path
    sf.write(path, audio_data, 16000)