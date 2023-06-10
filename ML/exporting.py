import os

import cv2
import imutils
import mediapipe as mp
from tqdm import tqdm

from saving import Savingdata


def main(media_name):
    mp_holistic = mp.solutions.holistic
    # cap = cv2.VideoCapture("http://192.168.33.100:8080//videofeed")
    cap = cv2.VideoCapture(media_name)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{media_name} can't be captured!")
                break
            frame = imutils.resize(frame, width=1000)

            im_height, im_width, _ = frame.shape

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = holistic.process(image)

            try:
                Savingdata(results, im_width, im_height)
            except Exception as e:
                print(f" Face in {media_name} not found!")


if __name__ == '__main__':
    folder_dir = "C:/Users/kubaz/PycharmProjects/PBL/ML/3dataset/NonDrowsy"
    names = os.listdir(folder_dir)

    for i in tqdm(range(len(names))):
        main(folder_dir + "/" + names[i])
