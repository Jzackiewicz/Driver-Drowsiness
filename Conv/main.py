import time

import cv2
import imutils

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


def make_detections(FAnalizer, PAnalizer, frame):
    FAnalizer.initialize_frame(frame)
    PAnalizer.initialize_frame(frame)

    if FAnalizer.landmarks_coords:  # Jeśli znaleziono twarz
        Outcomes.IS_FACE_DETECTED = True
        FAnalizer.get_all_ratios()
        FAnalizer.get_all_outcomes()
    else:
        Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

    if PAnalizer.landmarks_coords:  # Jeśli znaleziono ciało
        Outcomes.IS_POSE_DETECTED = True
        PAnalizer.get_all_ratios()
        PAnalizer.get_all_outcomes()

    else:
        Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała


def main(media_name):
    camera = cv2.VideoCapture(media_name)
    start_time = time.time()
    frame_counter = 0

    face_analyzer = FaceAnalysing()
    pose_analyzer = PoseAnalysing()

    read_thresholds_from_file()

    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=1000)  # max 1920 - min 1000

        make_detections(face_analyzer, pose_analyzer, frame)

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        inp = cv2.waitKey(1)  # using keyboard keys
        if inp in KEYS:
            exec(KEYS[inp])
        inp = None

        if face_analyzer.landmarks_coords:
            frame = face_analyzer.draw_indicators(Outcomes.ON_SCREEN)
        if pose_analyzer.landmarks_coords:
            frame = pose_analyzer.draw_indicators(Outcomes.ON_SCREEN)

        if Factors.CONF_MODE:
            configure_factors(frame)

        frame = show_fps(frame, fps)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    #video_dir = "..."
    video_dir = 0  # if using webcam
    main(video_dir)
