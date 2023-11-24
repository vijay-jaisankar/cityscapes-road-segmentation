"""
    Find and save the unique (i.e, not filtered by DHash) images for one-shot-synthesis algorithm
"""

import os
from imagededup.methods import DHash #noqa

# Set up the DHash algorithm
d_hasher = DHash()

# Load the images and set the hamming distance threshold
raw_images_path = "../data/processed/val/image"

"""
    Data Compression Schema
        - L: 10
        - M: 20
        - H: 30
"""
max_hamming_distance = 10

# Find duplicates to remove
duplicates_to_remove = d_hasher.find_duplicates_to_remove(
    image_dir = raw_images_path,
    max_distance_threshold = max_hamming_distance
)

# Find the unique images
all_images = os.listdir(raw_images_path)
unique_images = [f for f in all_images if f not in duplicates_to_remove]
unique_images.sort()
print(unique_images)

# Save the unique images
with open(f"unique_dhash_{max_hamming_distance}.txt", "w") as f:
    for img_file_name in unique_images:
        f.write(f"{img_file_name}\n")

