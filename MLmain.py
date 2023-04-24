import os
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm
import pickle
import numpy as np
from ML.saving import Savingdata


def main(media_name):
    with open('ML/yawn_noyawn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(model)

    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    cap = cv2.VideoCapture(media_name)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make Detections
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = holistic.process(image)
            #print(results.face_landmarks.landmark)
                # ------------------------------------ Saving2csv ------------------------------------
            # try:
            #     Savingdata(results)
            # except:
            #     pass


                #------------------------------------ Making detections ------------------------------------
            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                row = pose_row + face_row
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)
            except:
                pass

            cv2.imshow('Raw Webcam Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # names = os.listdir("C:/Users/kubaz/PycharmProjects/PBL/ML/0dataset/no_yawn")
    #
    # for i in tqdm(range(len(names))):
    #     main("C:/Users/kubaz/PycharmProjects/PBL/ML/0dataset/no_yawn/" + names[i])
    main("ziew.mp4")