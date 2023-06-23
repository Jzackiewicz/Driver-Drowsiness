import pickle

import cv2
import imutils
import mediapipe as mp
import numpy as np
import pandas as pd

from saving import Savingdata


def main(media_name, model_type):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(media_name)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:

        frame_counter = 0

        while True:
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                print(f"{media_name} can't be captured!")
                break

            frame = imutils.resize(frame, width=1080)

            im_height, im_width, _ = frame.shape

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = holistic.process(image)

            with open(model_type, 'rb') as f:
                model = pickle.load(f)

            side_length = 400
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

            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    models = ["models/model2NS.pkl", "models/model2AUGNS.pkl", "firstmodels/model3.pkl"]
    #video_dir = "..."
    video_dir = 0  # if using webcam

    main(video_dir, models[1])
