import torch
import os
import glob

def crop_spectrogram(input_tensor, target_time=5100):
    """
    Crop the mel spectrogram tensor to the target time steps.
    input_tensor: A tensor with shape [1, mel_bins, time_steps]
    target_time: The target number of time steps
    """
    # Check if the tensor is already the desired size
    if input_tensor.shape[1] == target_time:
        return input_tensor

    # Crop the spectrogram by selecting the first 'target_time' time steps
    cropped_tensor = input_tensor[:, :target_time]

    return cropped_tensor

def process_spectrogram_files(root, mode='train'):
    # Path to the folder containing the spectrograms
    files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.pt'))
    files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.pt'))

    # Target time steps for cropping the spectrograms
    target_time = 5100

    # Process files in both A and B folders
    for file_list in [files_A, files_B]:
        for file_path in file_list:
            # Load the mel spectrogram tensor
            spectrogram = torch.load(file_path)

            # Crop the spectrogram
            cropped_spectrogram = crop_spectrogram(spectrogram, target_time)

            # Save the cropped spectrogram back to the file
            torch.save(cropped_spectrogram, file_path)
            print(f"Cropped and saved: {file_path}")

if __name__ == "__main__":
    # Specify the root folder where your dataset is stored
    root = r'PyTorch-CycleGAN\datasets\country2rock'  # Update this path to your dataset folder

    # Process the training and testing data
    process_spectrogram_files(root, mode='train')
    process_spectrogram_files(root, mode='test')
