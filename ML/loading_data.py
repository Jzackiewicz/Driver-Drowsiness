import csv

import numpy as np

FILE_NAME = 'ML/coords.csv'
class_name = "dizzy"


class AIanalysing:
    def __init__(self, face_results, pose_results):
        self.face_landmarks = face_results.multi_face_landmarks[0].landmark
        self.pose_landmarks = pose_results.pose_landmarks.landmark
        # self.labels = self.get_labels()
        self.coords = self.get_all_coords()
        self.write2file()

    # def get_labels(self):
    #     coords_num = len(self.face_landmarks) + len(self.pose_landmarks)
    #     labels = ['class']
    #     for v in range(1, coords_num + 1):
    #         labels += ['x{}'.format(v), 'y{}'.format(v), 'z{}'.format(v), 'v{}'.format(v)]
    #     return labels

    def get_all_coords(self):
        pose_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in self.pose_landmarks]).flatten())
        face_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in self.face_landmarks]).flatten())

        return pose_row+face_row

    def write2file(self):
        with open(FILE_NAME, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = self.coords
            row.insert(0, class_name)
            csv_writer.writerow(row)
