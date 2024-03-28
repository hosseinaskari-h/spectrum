import librosa
import numpy as np
import matplotlib.pyplot as plt

def audio_to_spectrum_image(audio_path, output_path, figsize=(10, 4), cmap='gray'):
    # Loading the audio file
    y, sr = librosa.load(audio_path)
    # Calculate the (STFT)
    D = np.abs(librosa.stft(y))
    # Convert the magnitude to decibels
    DB = librosa.amplitude_to_db(D, ref=np.max)
    # Map the magnitudes to 0-255
    img_data = np.uint8(255 * (DB - DB.min()) / (DB.max() - DB.min()))
    # Plotting
    plt.figure(figsize=figsize)
    plt.imshow(img_data, aspect='auto', cmap=cmap, origin='lower')
    plt.axis('off')  # No axes
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage
audio_path = 'sam.wav'
output_path = 'output_spectrum_image.png'
audio_to_spectrum_image(audio_path, output_path)
