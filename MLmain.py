import os
import numpy as np
import pandas as pd
import cv2
import imutils
import mediapipe as mp
from tqdm import tqdm
import pickle
from ML.saving import Savingdata


def main(media_name):
    with open('ML/newhope_model.pkl', 'rb') as f:
        model = pickle.load(f)

    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    cap = cv2.VideoCapture(media_name)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                              refine_face_landmarks=True) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=1000)

            im_height, im_width, _ = frame.shape
            frame_ratio = im_width / im_height

            # Make Detections
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = holistic.process(image)

            # # print(results.face_landmarks.landmark)
            # # ------------------------------------ Saving2csv ------------------------------------
            # side_length = 500
            # #Savingdata(results, im_width, im_height)
            # try:
            #     Savingdata(results, im_width, im_height)  # Savingdata(results, im_width, im_height)
            # #     # dist, new_landmarks = saving.normalise()
            # #     # # square_ratio = 2 * dist / side_length
            # #     #
            # #     # nose_point = [round(results.face_landmarks.landmark[1].x * im_width),
            # #     #               round(results.face_landmarks.landmark[1].y * im_height)]
            # #     #
            # #     # frame = cv2.circle(frame, nose_point, 5, (0, 0, 255), -1)
            # #     # frame = cv2.circle(frame, nose_point, round(dist), (0, 0, 255), 5)
            # #     #
            # #     # start_point = (round(nose_point[0] - dist), round(nose_point[1] - dist))
            # #     # end_point = (round(nose_point[0] + dist), round(nose_point[1] + dist))
            # #     # frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 5)
            # #     #
            # #     # frame2 = np.zeros((side_length, side_length, 3), np.uint8)
            # #     # for landmark in results.face_landmarks.landmark:
            # #     #     iks = round(landmark.x * im_width)
            # #     #     igrek = round(landmark.y * im_height)
            # #     #     frame = cv2.circle(frame, (iks, igrek), 1, (0, 255, 0), -1)
            # #     #
            # #     # for landmark in new_landmarks:
            # #     #     frame2 = cv2.circle(frame2, (round(landmark[0] * side_length),
            # #     #                                  round(landmark[1] * side_length)), 1, (255, 255, 0), -1)
            # #     #     if landmark[0] == 0:
            # #     #         print(landmark[1])
            # #     #
            # #     # cv2.imshow('Face', frame2)
            # except:
            #     pass

            # ------------------------------------ Making detections ------------------------------------
            side_length = 500
            try:
                saving = Savingdata(results, im_width, im_height)
                ldmrks = list(np.array(saving.normalised_coords).flatten())

                X = pd.DataFrame([ldmrks])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                probability = max(body_language_prob) * 100
                frame = cv2.putText(frame, f"{body_language_class}: {round(probability)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                #print(body_language_class, body_language_prob)
                frame2 = np.zeros((side_length, side_length, 3), np.uint8)
                for landmark in saving.normalised_coords:
                    frame2 = cv2.circle(frame2, (round(landmark[0] * side_length),
                                                 round(landmark[1] * side_length)), 1, (255, 255, 0), -1)
                cv2.imshow('normalised view', frame2)
            except Exception as e:
                print(e)


            cv2.imshow('Raw Webcam Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # names = os.listdir("C:/Users/kubaz/PycharmProjects/PBL/ML/1dataset/FatigueSubjects")
    #
    # for i in tqdm(range(len(names))):
    #     main("C:/Users/kubaz/PycharmProjects/PBL/ML/1dataset/FatigueSubjects/" + names[i])
    main("nerwus.mp4")
