import pickle

import cv2
import imutils
import mediapipe as mp
import numpy as np
import pandas as pd

from saving import Savingdata


def main(media_name):
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    cap = cv2.VideoCapture("http://192.168.33.100:8080//videofeed")
    # cap = cv2.VideoCapture(media_name)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{media_name} can't be captured!")
                break

            frame = imutils.resize(frame, width=1080)

            im_height, im_width, _ = frame.shape

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = holistic.process(image)
            side_length = 400

            with open('model2.pkl', 'rb') as f:
                model = pickle.load(f)

            try:
                saving = Savingdata(results, im_width, im_height)
                ldmrks = list(np.array(saving.normalised_coords).flatten())

                X = pd.DataFrame([ldmrks])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                probability = max(body_language_prob) * 100
                frame = cv2.putText(frame, f"{body_language_class}: {round(probability)}%", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                frame2 = np.zeros((side_length, side_length, 3), np.uint8)
                for landmark in saving.normalised_coords:
                    frame2 = cv2.circle(frame2, (round(landmark[0] * side_length),
                                                 round(landmark[1] * side_length)), 1, (255, 255, 0), -1)
                cv2.imshow('normalised view', frame2)
            except Exception as e:
                print(e)

            # cv2.imwrite('org.jpg', frame)
            cv2.imshow('Input video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main("media/film3.mp4")
