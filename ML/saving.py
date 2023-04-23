import csv
import os

import numpy as np

FILE_NAME = 'ML/coordsnew.csv'
class_name = "Awake"

class Savingdata:
    def __init__(self, results):
        self.face_landmarks = results.face_landmarks.landmark
        self.pose_landmarks = results.pose_landmarks.landmark
        self.coords = self.get_all_coords()
        self.write2file()

    def get_all_coords(self):
        pose_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in self.pose_landmarks]).flatten())
        face_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in self.face_landmarks]).flatten())
        return pose_row+face_row

    def get_labels(self):
        num_coords = len(self.pose_landmarks) + len(self.face_landmarks)
        labels = ['class']
        for val in range(1, num_coords + 1):
            labels += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        with open(FILE_NAME, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(labels)

    def add_row(self):
        with open(FILE_NAME, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = self.coords
            row.insert(0, class_name)
            csv_writer.writerow(row)

    def write2file(self):
        if not os.path.exists(FILE_NAME):
            self.get_labels()
        self.add_row()

