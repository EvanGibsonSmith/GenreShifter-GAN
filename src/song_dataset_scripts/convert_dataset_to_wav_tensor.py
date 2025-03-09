import os
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Define the directory containing MP3 files
directory = r"PyTorch-CycleGAN\datasets\country2rock"  # Change this to your directory

# Walk through directory and process each MP3 file
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.mp3'):
            mp3_path = os.path.join(root, file)
            wav_tensor_path = os.path.join(root, f"{os.path.splitext(file)[0]}.pt")
            
            if not os.path.exists(wav_tensor_path):
                try:
                    waveform, sample_rate = torchaudio.load(mp3_path, normalize=True)
                    
                    # Resample to the easier to work with 16k
                    # Define resampler (from original sample rate to 16kHz)
                    target_sample_rate = 16000
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)

                    # Apply resampling
                    waveform_resampled = resampler(waveform)

                    # NOTE: Clip to the desired length to make training faster and easier
                    waveform_resampled = waveform_resampled[:, 0:100000]

                    torch.save(waveform_resampled, wav_tensor_path)
                    os.remove(mp3_path)
                    print(f"Replaced {file} with {wav_tensor_path}")
                
                except:
                    print(f"Failed to convert {file}")
                    os.remove(mp3_path)
            else:
                print(f"Skipped {file}. Path exists.")
