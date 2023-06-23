import csv
import math
import os

import numpy as np

FILE_NAME = 'new_dataset.csv'
CLASS_NAME = "clname"


class Savingdata:
    def __init__(self, results, width, height):
        self.face_landmarks = results.face_landmarks.landmark
        self.pose_landmarks = results.pose_landmarks.landmark
        self.image_width = width
        self.image_height = height

        self.normalise()
        self.write2file()

    def normalise(self):
        nose_point = [self.face_landmarks[1].x * self.image_width, self.face_landmarks[1].y * self.image_height,
                      self.face_landmarks[1].z * self.image_width]
        nose_dist = [math.sqrt((landmark.x * self.image_width - nose_point[0]) ** 2 +
                               (landmark.y * self.image_height - nose_point[1]) ** 2)
                     for landmark in self.face_landmarks]

        max_dist = max(nose_dist)


        square_side = max_dist * 2
        x_ratio = square_side / self.image_width
        y_ratio = square_side / self.image_height
        center_point = [max_dist, max_dist, max_dist]
        nose_center_offset = [abs(nose_point[0] - center_point[0]), abs(nose_point[1] - center_point[1]),
                              abs(nose_point[2] - center_point[2])]

        ldmrks = [[landmark.x * self.image_width - nose_center_offset[0],
                   landmark.y * self.image_height - nose_center_offset[1],
                   landmark.z * self.image_width - nose_center_offset[2]] for landmark in self.face_landmarks]

        normalised = [[landmark[0] / self.image_width / x_ratio, landmark[1] / self.image_height / y_ratio,
                       landmark[2] / self.image_width / x_ratio] for landmark in ldmrks]

        self.normalised_coords = normalised

        return max_dist, normalised

    def get_labels(self):
        num_coords = len(self.normalised_coords)
        labels = ['class']
        for val in range(1, num_coords + 1):
            labels += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
        with open(FILE_NAME, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(labels)

    def add_row(self):
        with open(FILE_NAME, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(np.array(self.normalised_coords).flatten())
            row.insert(0, CLASS_NAME)
            csv_writer.writerow(row)

    def write2file(self):
        if not os.path.exists(FILE_NAME):
            self.get_labels()
        self.add_row()
