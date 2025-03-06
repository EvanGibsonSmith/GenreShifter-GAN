from PIL import Image
import os

def check_rgb_images(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Only process jpg images
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Check if the image is in RGB mode
                    if img.mode != 'RGB':
                        print(f"Image '{filename}' is not in RGB mode, it is in {img.mode} mode.")
            except Exception as e:
                print(f"Error opening image {filename}: {e}")

# Provide the folder path here
folder_path = r'PyTorch-CycleGAN\datasets\horse2zebra\train\B'  # Replace with the actual folder path
check_rgb_images(folder_path)
print("Done checking images")
