import imgaug.augmenters as iaa
import cv2
import glob

images = []
images_path = glob.glob("C:/Users/Aleksandra/Documents/Studia/Sem6/Drowsiness/PBL/zdjecia/*.jpg")

path = r"C:/Users/Aleksandra/Documents/Studia/Sem6/Drowsiness/PBL/zdjecia/test/"

for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)


augmentation_flip = iaa.Sequential([
    iaa.Fliplr(0.5),
])
augmentation_multiply = iaa.Sequential([
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
])
augmentation_contrast = iaa.Sequential([
    iaa.LinearContrast((0.75, 1.5)),
])
augmentation_gaussian = iaa.Sequential([
    iaa.GaussianBlur((0.0, 3.0)),
])

i = 0
#while (i<4600):
#while True:

augemented_images1 = augmentation_flip(images=images)
augemented_images2 = augmentation_multiply(images=images)
augemented_images3 = augmentation_contrast(images=images)
augemented_images4 = augmentation_gaussian(images=images)
augemented_images = augemented_images1+augemented_images2+augemented_images3+augemented_images4
for img in augemented_images:
    #cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.imwrite(path + str(i) + 'test1.jpg', img)
    i += 1
#if i == 10
#    break