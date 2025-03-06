import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/horse2zebra-dataset")

print("Path to dataset files:", path)