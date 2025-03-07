import os
import pandas as pd
import shutil
import random

# Specify your paths
mp3_folder = 'fma_small'  # Folder containing mp3 files
csv_file = 'mp3_titles_and_genres_small.csv'  # CSV file containing the song names and genres
output_folder = 'PyTorch-CycleGAN\datasets\country2rock'  # Folder where the train/test structure will be created

# Read the CSV file and create a dictionary of songs with their corresponding genre
df = pd.read_csv(csv_file)

genre1, genre2 = 'Folk', 'Rock'
songs_by_genre = {
    genre1: list(df[df['genre_str'] == genre1]['path']),
    genre2: list(df[df['genre_str'] == genre2]['path'])
}

# Create the train/test structure
os.makedirs(os.path.join(output_folder, 'train/A'), exist_ok=True)  # For genre1 songs in train
os.makedirs(os.path.join(output_folder, 'train/B'), exist_ok=True)  # For genre2 songs in train
os.makedirs(os.path.join(output_folder, 'test/A'), exist_ok=True)   # For genre1 songs in test
os.makedirs(os.path.join(output_folder, 'test/B'), exist_ok=True)   # For genre2 songs in test

# Function to move files to their respective folders
def move_files(file_list, genre, set_type):
    for file in file_list:
        source_path = os.path.join(mp3_folder, file)
        if os.path.exists(source_path):
            destination_folder = os.path.join(output_folder, f'{set_type}', genre, os.path.basename(file))
            print(source_path, destination_folder)
            shutil.copy(source_path, destination_folder)

# Randomly split the songs into train and test sets (90% train, 10% test)
for idx, genre in enumerate(songs_by_genre):

    random.shuffle(songs_by_genre[genre])  # Shuffle the list of songs
    test_size = int(len(songs_by_genre[genre]) * 0.1)  # 10% for test
    test_songs = songs_by_genre[genre][:test_size]
    train_songs = songs_by_genre[genre][test_size:]

    if (idx==0):
        genre_folder = 'A'
    else:
        genre_folder = 'B'

    # Move files into the respective folders
    move_files(test_songs, genre_folder, 'test')
    move_files(train_songs, genre_folder, 'train')

print("Files have been successfully moved!")
