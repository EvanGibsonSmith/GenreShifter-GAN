import torch

#spec = torch.load(r"PyTorch-CycleGAN\datasets\country2rock\train\A\000890.pt")
#print(spec.shape)

# For wav
import torchaudio

waveform, sample_rate = torchaudio.load(r'PyTorch-CycleGAN\datasets\country2rock\test\A\000212.wav')
print(waveform.shape)
