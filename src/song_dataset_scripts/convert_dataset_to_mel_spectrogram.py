import os
import torch
import torchaudio
from pydub import AudioSegment
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Define the directory containing MP3 files
directory = r"PyTorch-CycleGAN\datasets\country2rock"  # Change this to your directory

# Function to convert MP3 to WAV
def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Walk through directory and process each MP3 file
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.mp3'):
            mp3_path = os.path.join(root, file)
            wav_path = os.path.join(root, f"{os.path.splitext(file)[0]}.wav")
            
            # Convert MP3 to WAV
            mp3_to_wav(mp3_path, wav_path)
            
            # Load the WAV file
            signal, rate = torchaudio.load(wav_path)
            
            # If stereo, convert to mono by averaging the two channels
            if signal.shape[0] == 2:  # Stereo
                signal = signal.mean(dim=0, keepdim=True)  # Average the two channels to make it mono

            # Compute the mel spectrogram
            spectrogram, _ = mel_spectogram(
                audio=signal.squeeze(),
                sample_rate=22050,
                hop_length=256,
                win_length=None,
                n_mels=80,
                n_fft=1024,
                f_min=0.0,
                f_max=8000.0,
                power=1,
                normalized=False,
                min_max_energy_norm=True,
                norm="slaney",
                mel_scale="slaney",
                compression=True
            )

            # Save the spectrogram as a .pt file, replacing the original MP3
            spectrogram_path = os.path.join(root, f"{os.path.splitext(file)[0]}.pt")
            torch.save(spectrogram, spectrogram_path)
            
            # Remove the original MP3 file
            os.remove(mp3_path)
            
            # Optionally, remove the WAV file after processing (if you don't need it)
            os.remove(wav_path)
            
            print(f"Replaced {file} with {spectrogram_path}")
