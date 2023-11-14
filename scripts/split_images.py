"""
    Collection of utilities to split and save the halves into two folders.
    @note The format of Cityscapes' images is [IMG][MASK]
"""
import cv2
import os
from tqdm import tqdm


"""
    @method get_all_images_dir
        Extract the file names of all images (*.jpg) from a directory
"""
def get_all_images_dir(dir_path):
    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            # Check if the file is a .jpg file
            if ".jpg" in path:
                res.append(path)
    
    return res


"""
    @method split_image_half
        Split the image provided in half and return both halves as `cv2` images
"""
def split_image_half(img):
    # Find the width cutoff
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2

    # Split the images in two and return them
    scene = img[:, :width_cutoff]
    mask = img[:, width_cutoff:]

    return scene, mask

if __name__ == "__main__":
    # Make folders for the processed images for scenes and masks
    try:
        os.makedirs("../data/processed/train/image")
        os.makedirs("../data/processed/train/mask")
        os.makedirs("../data/processed/val/image")
        os.makedirs("../data/processed/val/mask")
    except FileExistsError as e:
        print(f"Directories already created: {e}")
        pass

    # Generate images and masks for the train set
    train_path = "../data/cityscapes_data/train"
    train_merged_images = get_all_images_dir(train_path)

    for image_file in tqdm(train_merged_images):
        img = cv2.imread(f"{train_path}/{image_file}")
        scene, mask = split_image_half(img)

        # Save the images and masks
        cv2.imwrite(f"../data/processed/train/image/{image_file}", scene)
        cv2.imwrite(f"../data/processed/train/mask/{image_file}", mask)

    # Generate images and masks for the validation set
    val_path = "../data/cityscapes_data/val"
    val_merged_images = get_all_images_dir(val_path)

    for image_file in tqdm(val_merged_images):
        img = cv2.imread(f"{val_path}/{image_file}")
        scene, mask = split_image_half(img)

        # Save the images and masks
        cv2.imwrite(f"../data/processed/val/image/{image_file}", scene)
        cv2.imwrite(f"../data/processed/val/mask/{image_file}", mask)
