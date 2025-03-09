import torch
import soundfile as sf
import sounddevice as sd

# Load the tensor (assuming you have the .pt file with audio tensor)
# This is scary at 22050: PyTorch-CycleGAN\datasets\country2rock\test\A\005879.pt
# r"PyTorch-CycleGAN\datasets\country2rock\test\A\000140.pt"
# Cool minor folk r"PyTorch-CycleGAN\datasets\country2rock\test\A\110439.pt"
# Another cool guitar r"PyTorch-CycleGAN\datasets\country2rock\test\A\067016.pt"
waveform = torch.load(r"PyTorch-CycleGAN\datasets\country2rock\train\A\007709.pt")
waveform = waveform.numpy()
print(waveform.shape)
sample_rate = 16000 # Need this
sd.play(waveform.T, sample_rate)  
sd.wait()
