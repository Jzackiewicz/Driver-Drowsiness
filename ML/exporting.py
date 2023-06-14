import os

import cv2
import imutils
import mediapipe as mp
from tqdm import tqdm

from saving import Savingdata


def add_borders(image):
    image = imutils.resize(image, width=500)
    border_size = 100
    image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return image


def main(media_name):
    frame = cv2.imread(media_name)
    # frame = add_borders(frame) # Needed if images are too small

    im_height, im_width, _ = frame.shape

    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = holistic.process(image)

        try:
            Savingdata(results, im_width, im_height)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    data_dir = "xyz"
    names = os.listdir(data_dir)

    for i in tqdm(range(len(names))):
        main(data_dir + "/" + names[i])
