import cv2
import mediapipe as mp
import numpy as np

import utils
from base import MediaPipeAnalysing
from database import Indices, Factors, Counters, Outcomes


class FaceAnalysing(MediaPipeAnalysing):
    def __init__(self):
        super().__init__('Face')
        self.solution = 'Face'
        self.head_angles = None
        self.lips_ratio = None
        self.eyes_ratio = None
        self.looking_direction = None

    def estimate_lips_ratio(self, draw=False):
        upper_indices = Indices.UPPER_LIP
        lower_indices = Indices.LOWER_LIP
        upper_lip = self.landmarks_coords[upper_indices[13]]
        lower_lip = self.landmarks_coords[lower_indices[16]]
        left_corner = self.landmarks_coords[lower_indices[0]]
        right_cornet = self.landmarks_coords[lower_indices[10]]

        vertical_distance = self.get_euclidean_distance(upper_lip, lower_lip)
        horizontal_distance = self.get_euclidean_distance(left_corner, right_cornet)

        ratio = vertical_distance / horizontal_distance

        if draw:
            cv2.line(self.img, left_corner, right_cornet, utils.MAGENTA, 2)
            cv2.line(self.img, upper_lip, lower_lip, utils.RED, 2)

        return ratio

    def detect_yawning(self):
        if self.lips_ratio > Factors.LIPS_RATIO_FACTOR:
            Outcomes.IS_MOUTH_OPEN = True
            Counters.OPENED_MOUTH_COUNTER += 1
            if Counters.OPENED_MOUTH_COUNTER > Factors.OPENED_MOUTH_FRAME:
                return True
        else:
            Counters.OPENED_MOUTH_COUNTER = 0
            Outcomes.IS_MOUTH_OPEN = False
            return False

    def estimate_eyes_ratio(self, *draw):
        right_indices = Indices.RIGHT_EYE
        left_indices = Indices.LEFT_EYE
        rh_right = self.landmarks_coords[right_indices[0]]
        rh_left = self.landmarks_coords[right_indices[8]]
        rv_top = self.landmarks_coords[right_indices[12]]
        rv_bottom = self.landmarks_coords[right_indices[4]]

        lh_right = self.landmarks_coords[left_indices[0]]
        lh_left = self.landmarks_coords[left_indices[8]]
        lv_top = self.landmarks_coords[left_indices[12]]
        lv_bottom = self.landmarks_coords[left_indices[4]]

        rh_distance = self.get_euclidean_distance(rh_right, rh_left)
        rv_distance = self.get_euclidean_distance(rv_top, rv_bottom)

        lv_distance = self.get_euclidean_distance(lv_top, lv_bottom)
        lh_distance = self.get_euclidean_distance(lh_right, lh_left)

        r_ratio = rv_distance / rh_distance
        l_ratio = lv_distance / lh_distance

        ratio = (r_ratio + l_ratio) / 2
        Outcomes.EYES_RATIO.append(ratio)

        if draw:
            cv2.line(self.img, rh_right, rh_left, utils.MAGENTA, 2)
            cv2.line(self.img, rv_top, rv_bottom, utils.RED, 2)

        return ratio

    def get_eyes_state(self):
        if self.eyes_ratio < Factors.EYES_RATIO_FACTOR:
            Counters.CLOSED_EYES_COUNTER += 1
            return True
        else:
            Counters.CLOSED_EYES_COUNTER = 0
            return False

    def estimate_eye_position(self, eye_iris_indices, eye_mesh_indices, draw=False):
        eye_pupil_center = self.get_average_point(eye_iris_indices)
        eye_center = self.get_average_point(eye_mesh_indices)

        eye_left_border = self.landmarks_coords[eye_mesh_indices[0]]
        eye_right_border = self.landmarks_coords[eye_mesh_indices[8]]
        eye_top_border = self.landmarks_coords[eye_mesh_indices[-1]]
        eye_bottom_border = self.landmarks_coords[eye_mesh_indices[-2]]

        h_pupil_distance = eye_center[0] - eye_pupil_center[0]
        v_pupil_distance = eye_center[1] - eye_pupil_center[1]

        pupil_direction_ratio = [h_pupil_distance, v_pupil_distance]

        if draw:
            cv2.circle(self.img, eye_left_border, 2, (0, 0, 255), -1)
            cv2.circle(self.img, eye_right_border, 2, (0, 0, 255), -1)
            cv2.circle(self.img, eye_top_border, 2, (0, 0, 255), -1)
            cv2.circle(self.img, eye_bottom_border, 2, (0, 0, 255), -1)

            eye_center = self.get_average_point(eye_mesh_indices)
            cv2.circle(self.img, eye_pupil_center, 4, (0, 0, 255), -1)
            cv2.circle(self.img, eye_center, 4, (0, 255, 0), -1)
            cv2.line(self.img, eye_center, eye_pupil_center, (0, 255, 255), 3)
        return pupil_direction_ratio

    def get_looking_direction(self):
        h_ratio = self.looking_direction[0]
        v_ratio = self.looking_direction[1]

        if h_ratio < Factors.PUPIL_LEFT_THRESHOLD:
            looking_direction = "Left"
        elif h_ratio > Factors.PUPIL_RIGHT_THRESHOLD:
            looking_direction = "Right"
        elif v_ratio > Factors.PUPIL_UP_THRESHOLD:
            looking_direction = "Up"
        elif v_ratio < Factors.PUPIL_DOWN_THRESHOLD:
            looking_direction = "Down"
        else:
            looking_direction = "Forward"

        return looking_direction

    def estimate_head_position_angles(self, draw=False):
        face_3d = []
        face_2d = []
        position = []
        for face_landmarks in self.results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * self.frame_width, lm.y * self.frame_height)

                    x, y = int(lm.x * self.frame_width), int(lm.y * self.frame_height)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * self.frame_width

            cam_matrix = np.array([[focal_length, 0, self.frame_height / 2],
                                   [0, focal_length, self.frame_width / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            position.append(angles[0] * 360)
            position.append(angles[1] * 360)

            if draw:
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + position[0] * 10), int(nose_2d[1] - position[1] * 10))
                cv2.line(self.img, p1, p2, (255, 0, 0), 3)

        return position

    def get_head_position(self):
        if self.head_angles[1] > Factors.HEAD_LEFT_THRESHOLD:
            direction = "Left"
        elif self.head_angles[1] < Factors.HEAD_RIGHT_THRESHOLD:
            direction = "Right"
        elif self.head_angles[0] < Factors.HEAD_DOWN_THRESHOLD:
            direction = "Down"
        elif self.head_angles[0] > Factors.HEAD_UP_THRESHOLD:
            direction = "Up"
        else:
            direction = "Forward"

        return direction

    def get_all_ratios(self):
        Outcomes.EYES_RATIO.append(self.estimate_eyes_ratio())  # Stosunki odcinków oka
        Outcomes.MOUTH_RATIO.append(self.estimate_lips_ratio())  # Stosunki odcinków ust
        Outcomes.HEAD_POSITION_ANGLES.append(self.estimate_head_position_angles())  # Kąty obrotów głowy

        left_eye_position = self.estimate_eye_position(Indices.RIGHT_IRIS, Indices.RIGHT_EYE)
        right_eye_position = self.estimate_eye_position(Indices.LEFT_IRIS, Indices.LEFT_EYE)
        both_eyes_position = [(left_eye_position[0] + right_eye_position[0]) / 2,
                              (left_eye_position[1] + right_eye_position[1]) / 2]
        Outcomes.PUPILS_DIRECTION_RATIO.append(both_eyes_position) # Średnie odległości źrenic od środka oka

    def get_all_outcomes(self):
        self.eyes_ratio = sum(Outcomes.EYES_RATIO) / len(Outcomes.EYES_RATIO)
        Outcomes.ARE_EYES_CLOSED = self.get_eyes_state()
        self.eyes_ratio = round(self.eyes_ratio, 2)
        Outcomes.EYES_RATIO = []
        Outcomes.EYES_RATIO.append(self.eyes_ratio)

        angles = Outcomes.HEAD_POSITION_ANGLES
        self.head_angles = list(map(lambda *x: sum(x) / len(angles), *angles))
        Outcomes.HEAD_POSITION = self.get_head_position()
        self.head_angles = [round(x, 1) for x in self.head_angles]
        Outcomes.HEAD_POSITION_ANGLES = []
        Outcomes.HEAD_POSITION_ANGLES.append(self.head_angles)

        directions = Outcomes.PUPILS_DIRECTION_RATIO
        self.looking_direction = list(map(lambda *x: sum(x) / len(directions), *directions))
        Outcomes.LOOKING_DIRECTION = self.get_looking_direction()
        self.looking_direction = [round(x, 1) for x in self.looking_direction]
        Outcomes.PUPILS_DIRECTION_RATIO = []
        Outcomes.PUPILS_DIRECTION_RATIO.append(self.looking_direction)

        self.lips_ratio = sum(Outcomes.MOUTH_RATIO) / len(Outcomes.MOUTH_RATIO)
        Outcomes.IS_YAWNING = self.detect_yawning()
        self.lips_ratio = round(self.lips_ratio, 2)
        Outcomes.MOUTH_RATIO = []
        Outcomes.MOUTH_RATIO.append(self.lips_ratio)

    def draw_indicators(self, objects):
        height = 150
        f_width = round(1400 / 2100 * self.frame_width)
        f_height = round(1000 / 1181 * self.frame_height)
        font_scale = self.frame_width / 2100 * 1.5

        if 'face mesh' in objects:
            mp_drawing = mp.solutions.drawing_utils
            mp_face_mesh = mp.solutions.face_mesh
            draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=utils.BLUE)

            if self.results.multi_face_landmarks:
                for face_landmarks in self.results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(self.img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, draw_spec,
                                              draw_spec)

        if 'eyes' in objects:
            height -= 150
            self.estimate_eyes_ratio(True)
            if Outcomes.ARE_EYES_CLOSED:
                color = utils.RED
            else:
                color = utils.WHITE

            utils.colorBackgroundText(self.img, f'Eyes ratio: {self.eyes_ratio}', Factors.FONTS, font_scale,
                                      (f_width, f_height + height), 2, utils.BLACK, color)
            utils.colorBackgroundText(self.img, f'Threshold: {Factors.EYES_RATIO_FACTOR}', Factors.FONTS, font_scale,
                                      (f_width, f_height + 50 + height), 2, utils.BLACK, utils.WHITE)

        if 'pupils' in objects:
            height -= 150

            self.estimate_eye_position(Indices.RIGHT_IRIS, Indices.RIGHT_EYE, True)
            self.estimate_eye_position(Indices.LEFT_IRIS, Indices.LEFT_EYE, True)

            if Outcomes.LOOKING_DIRECTION == 'Right':
                color = utils.YELLOW
                pos = 'R|'
            elif Outcomes.LOOKING_DIRECTION == 'Left':
                color = utils.ORANGE
                pos = 'L|'
            elif Outcomes.LOOKING_DIRECTION == 'Up':
                color = utils.BLUE
                pos = 'U|'
            elif Outcomes.LOOKING_DIRECTION == 'Down':
                color = utils.PURPLE
                pos = 'D|'
            else:
                color = utils.WHITE
                pos = ''

            utils.colorBackgroundText(self.img,
                                      f'Eyes pos.: {pos}{self.looking_direction[0]}|{self.looking_direction[1]}',
                                      Factors.FONTS, font_scale,
                                      (f_width, f_height + height), 2, utils.BLACK, color)
            utils.colorBackgroundText(self.img,
                                      f'Threshold: {Factors.PUPIL_LEFT_THRESHOLD}|{Factors.PUPIL_RIGHT_THRESHOLD}|'
                                      f'{Factors.PUPIL_UP_THRESHOLD}|{Factors.PUPIL_DOWN_THRESHOLD}',
                                      Factors.FONTS, font_scale,
                                      (f_width, f_height + 50 + height), 2, utils.BLACK, utils.WHITE)

        if 'lips' in objects:
            height -= 150
            self.estimate_lips_ratio(True)

            if Outcomes.IS_MOUTH_OPEN:
                color = utils.RED
            else:
                color = utils.WHITE

            utils.colorBackgroundText(self.img, f'Lips ratio: {self.lips_ratio}', Factors.FONTS, font_scale,
                                      (f_width, f_height + height), 2, utils.BLACK, color)
            utils.colorBackgroundText(self.img, f'Threshold: {Factors.LIPS_RATIO_FACTOR}', Factors.FONTS, font_scale,
                                      (f_width, f_height + 50 + height), 2, utils.BLACK, utils.WHITE)

        if 'head' in objects:
            height -= 150
            self.estimate_head_position_angles(True)

            if Outcomes.HEAD_POSITION == 'Right':
                color = utils.YELLOW
                pos = 'R|'
            elif Outcomes.HEAD_POSITION == 'Left':
                color = utils.ORANGE
                pos = 'L|'
            elif Outcomes.HEAD_POSITION == 'Up':
                color = utils.BLUE
                pos = 'U|'
            elif Outcomes.HEAD_POSITION == 'Down':
                color = utils.PURPLE
                pos = 'D|'
            else:
                color = utils.WHITE
                pos = ''

            utils.colorBackgroundText(self.img, f'Head pos.: {pos}{self.head_angles[1]}|{self.head_angles[0]}',
                                      Factors.FONTS, font_scale, (f_width, f_height + height), 2, utils.BLACK, color)
            utils.colorBackgroundText(self.img,
                                      f'Threshold:'
                                      f'{Factors.HEAD_LEFT_THRESHOLD}|{Factors.HEAD_RIGHT_THRESHOLD}|'
                                      f'{Factors.HEAD_UP_THRESHOLD}|{Factors.HEAD_DOWN_THRESHOLD}',
                                      Factors.FONTS, font_scale, (f_width, f_height + 50 + height), 2, utils.BLACK,
                                      utils.WHITE)

        return self.img
