"""
    Localise the road from the CITYSCAPES images' masks.
"""

import cv2
import numpy as np
from tqdm import tqdm
from split_images import get_all_images_dir
import os

# RGB colours for `road` and `other`
road_rgb = np.array([127, 63, 128])
other_rgb = np.array([0, 0, 0])
white_rgb = np.array([255, 255, 255])

"""
    @method recolour_image
        Recolour image based on the binary segmentation conditions
"""
def recolour_image(img, road_rgb = road_rgb, other_rgb = other_rgb):
    # Loop through the image
    height = img.shape[0]
    width = img.shape[1]

    # Make the non road pixels `other_rgb`
    for x in range(width):
        for y in range(height):
            current_pixel = (img[y][x])
            if np.array_equal(current_pixel, road_rgb) is False:
                img[y][x] = other_rgb
            else:
                img[y][x] = white_rgb

    # Return the updated image
    return img

if __name__ == "__main__":
    # Make folders for the processed images for scenes and masks
    try:
        os.makedirs("../data/processed/train/binary_road_mask")
        os.makedirs("../data/processed/val/binary_road_mask")
    except FileExistsError as e:
        print(f"Directories already created: {e}")
        pass

    # Generate road masks for the train set
    train_path = "../data/processed/train/mask"
    train_merged_images = get_all_images_dir(train_path)

    for image_file in tqdm(train_merged_images):
        img = cv2.imread(f"{train_path}/{image_file}")
        road_img = recolour_image(img)

        # Save the road masks
        cv2.imwrite(f"../data/processed/train/binary_road_mask/{image_file}", road_img)


    # Generate road masks for the validation set
    val_path = "../data/processed/val/mask"
    val_merged_images = get_all_images_dir(val_path)

    for image_file in tqdm(val_merged_images):
        img = cv2.imread(f"{val_path}/{image_file}")
        road_img = recolour_image(img)

        # Save the road masks
        cv2.imwrite(f"../data/processed/val/binary_road_mask/{image_file}", road_img)
