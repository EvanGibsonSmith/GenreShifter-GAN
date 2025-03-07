import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import librosa
import librosa.display
import numpy as np
import sounddevice as sd

# Assuming you have the mel spectrogram in 'mel_spec'
spectrogram_path = r"PyTorch-CycleGAN\output\country_to_rock\fakeB\0009.pt"  # Change this to your .pt file path
mel_spec = torch.load(spectrogram_path)  # Load your mel spectrogram

# Convert mel spectrogram to numpy
mel_spec = mel_spec.squeeze().cpu().numpy()  # Convert to 2D numpy array if it's 3D

# Parameters (should match those used for the spectrogram generation)
n_mels = 80
n_fft = 1024
hop_length = 256
sample_rate = 22050
f_min = 0.0
f_max = 8000.0

# Inverse mel spectrogram -> Linear spectrogram (use librosa)
# Reconstruct the linear spectrogram using the inverse mel scale
mel_to_linear = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sample_rate, n_fft=n_fft, power=1.0, fmin=f_min, fmax=f_max)

# Reconstruct waveform from linear spectrogram using Griffin-Lim (or any other method)
waveform = librosa.istft(mel_to_linear, hop_length=hop_length)


# Display using imshow
plt.imshow(mel_spec, cmap='viridis')  # Use cmap='gray' for grayscale
plt.colorbar()  # Optional: adds a colorbar
plt.show()


# Play the waveform
print("Playing...")
sd.play(waveform, sample_rate)
sd.wait()
