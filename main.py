import time

import cv2
import imutils

import utils
from database import Factors, Outcomes, KEYS
from face import FaceAnalysing
from pose import PoseAnalysing
from thresholdmenagement import *


def show_fps(frame, fps):
    frame_height, frame_width = frame.shape[:2]
    font_scale = frame_width / 1700
    frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', Factors.FONTS, font_scale,
                                     (round(frame_width / 100), round(frame_height / 20)), bgOpacity=0.9,
                                     textThickness=2)

    utils.colorBackgroundText(frame, f'Face : {Outcomes.IS_FACE_DETECTED}', Factors.FONTS, font_scale * 0.8,
                              (round(frame_width / 100), round(frame_height / 10)), 2, utils.BLACK, utils.WHITE)
    utils.colorBackgroundText(frame, f'Pose : {Outcomes.IS_POSE_DETECTED}', Factors.FONTS, font_scale * 0.8,
                              (round(frame_width / 100), round(frame_height * 2 / 15)), 2, utils.BLACK, utils.WHITE)

    return frame


def main():
    camera = cv2.VideoCapture("eniu.mp4")

    start_time = time.time()
    frame_counter = 0

    i = 0
    face_analyzer = FaceAnalysing()
    pose_analyzer = PoseAnalysing()

    read_thresholds_from_file()

    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=1400)  # max 1920 - min 1000
        if frame_counter % Factors.OPTIMIZATION_FACTOR == 0:  # Wpływ na optymalizację kodu -> pomijanie klatek
            i += 1
            face_analyzer.initialize_frame(frame)
            pose_analyzer.initialize_frame(frame)

            if Factors.AVERAGING_FACTOR == 1:  # ---------Brak uśredniania wyników -> chaotyczne wartości-----------
                if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                    Outcomes.IS_FACE_DETECTED = True
                    face_analyzer.get_all_ratios()
                    face_analyzer.get_all_outcomes()

                else:
                    Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

                if pose_analyzer.landmarks_coords:  # Jeśli znaleziono ciało
                    Outcomes.IS_POSE_DETECTED = True
                    pose_analyzer.get_all_ratios()
                    pose_analyzer.get_all_outcomes()
                else:
                    Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała

            else:
                if i % Factors.AVERAGING_FACTOR != 0:  # ---------Uśrednianie wyników-----------

                    if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                        Outcomes.IS_FACE_DETECTED = True
                        face_analyzer.get_all_ratios()

                    else:
                        Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

                    if pose_analyzer.landmarks_coords:  # Jeśli znaleziono ciało
                        Outcomes.IS_POSE_DETECTED = True
                        pose_analyzer.get_all_ratios()

                    else:
                        Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała
                else:
                    if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                        Outcomes.IS_FACE_DETECTED = True
                        face_analyzer.get_all_ratios()
                        face_analyzer.get_all_outcomes()

                    else:
                        Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

                    if pose_analyzer.landmarks_coords:  # Jeśli znaleziono ciało
                        Outcomes.IS_POSE_DETECTED = True
                        pose_analyzer.get_all_ratios()
                        pose_analyzer.get_all_outcomes()

                    else:
                        Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        inp = cv2.waitKey(1)
        if inp in KEYS:
            exec(KEYS[inp])
        inp = None

        if face_analyzer.landmarks_coords:
            frame = face_analyzer.draw_indicators(Outcomes.ON_SCREEN)
        if pose_analyzer.landmarks_coords:
            frame = pose_analyzer.draw_indicators(Outcomes.ON_SCREEN, frame)

        if Factors.CONF_MODE:
            configure_factors(frame)

        frame = show_fps(frame, fps)

        cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        cv2.imwrite('wynik1.jpg', frame)

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()
