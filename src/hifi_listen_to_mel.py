import torch
from speechbrain.inference.vocoders import HIFIGAN
import sounddevice as sd

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")


# Assuming you have the mel spectrogram in 'mel_spec'
print("Loading Spectrogram")
spectrogram_path = r"PyTorch-CycleGAN\output\country_to_rock\fakeB\0155.pt"  # Change this to your .pt file path
mel_spec = torch.load(spectrogram_path)  # Load your mel spectrogram

# Convert mel spectrogram to numpy
mel_spec = mel_spec.squeeze()  # Convert to 2D numpy array if it's 3D
mel_spec = mel_spec[:, :1000] # Clip size to be short for model
mel_spec = mel_spec.unsqueeze(0) # "Batch" of 1

print("Processing Waveform with Hifi GAN")
waveform = hifi_gan.decode_batch(mel_spec)
waveform = waveform.squeeze()

print("Playing Waveform")
sd.play(waveform, 22050)
sd.wait()

