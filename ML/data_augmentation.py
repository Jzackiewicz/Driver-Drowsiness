import os

import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm

from_path = "..."
to_path = r"..."


def get_new_images(image_dir):
    base_img = cv2.imread(image_dir)
    augmentation_flip = iaa.Sequential(iaa.Fliplr(1))
    augmentation_multiply = iaa.Sequential(iaa.Multiply((1, 3)))
    augmentation_contrast = iaa.Sequential(iaa.LinearContrast((0.7, 3)))
    augmentation_gaussian = iaa.Sequential(iaa.GaussianBlur(sigma=(3, 6)))

    new_img1 = augmentation_flip(images=[base_img])
    new_img2 = augmentation_multiply(images=[base_img])
    new_img3 = augmentation_contrast(images=[base_img])
    new_img4 = augmentation_gaussian(images=[base_img])

    new_images = [new_img1, new_img2, new_img3, new_img4]
    return new_images


def write_new_images(images, idx):
    j = 0
    for img in images:
        cv2.imwrite(to_path + 'pic' + str(idx + 1) + '_aug(' + str(j + 1) + ').jpg', img[0])
        j += 1


if __name__ == '__main__':
    names = os.listdir(from_path)
    i = 0
    for name in tqdm(range(len(names))):
        augmented_images = get_new_images(from_path + '/' + str(names[name]))
        write_new_images(augmented_images, i)
        i += 1
