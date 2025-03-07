import torch

# Load the .pt file
data = torch.load('genre_shifter_tensors_lite\spectrogram_tensors_lite.pt')

# Inspect the loaded data
print(data.shape)  # Check the type of the object

# If it's a dictionary, print its keys (common for checkpoints)
if isinstance(data, dict):
    print("Keys in the dataset:", data.keys())

# If it's a tensor or another structure, print it
else:
    print(data)
