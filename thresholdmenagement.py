from database import Factors, Outcomes, KEYS


def add_remove(li, elem):
    if elem not in li:
        li.append(elem)
    else:
        li.remove(elem)
    return li


def write_thresholds_2file():
    f = open("thresholds.txt", "w")
    factors = [str(Factors.EYES_RATIO_FACTOR), str(Factors.LIPS_RATIO_FACTOR), str(Factors.HAND_FACE_DISTANCE_FACTOR),
               str(Factors.PUPIL_LEFT_THRESHOLD), str(Factors.PUPIL_RIGHT_THRESHOLD), str(Factors.PUPIL_UP_THRESHOLD),
               str(Factors.PUPIL_DOWN_THRESHOLD),
               str(Factors.HEAD_LEFT_THRESHOLD), str(Factors.HEAD_RIGHT_THRESHOLD), str(Factors.HEAD_UP_THRESHOLD),
               str(Factors.HEAD_DOWN_THRESHOLD)]
    for line in factors:
        f.write(line)
        f.write("\n")
    f.close()


def read_thresholds_from_file():
    f = open("thresholds.txt", "r")
    new_factors = []
    for line in f:
        new_factor = line.strip("\n")
        new_factors.append([new_factor])
    print(len(new_factors))
    Factors.EYES_RATIO_FACTOR = float(new_factors[0][0])
    Factors.LIPS_RATIO_FACTOR = float(new_factors[1][0])
    Factors.HAND_FACE_DISTANCE_FACTOR = float(new_factors[2][0])

    Factors.PUPIL_LEFT_THRESHOLD = float(new_factors[3][0])
    Factors.PUPIL_RIGHT_THRESHOLD = float(new_factors[4][0])
    Factors.PUPIL_UP_THRESHOLD = float(new_factors[5][0])
    Factors.PUPIL_DOWN_THRESHOLD = float(new_factors[6][0])

    Factors.HEAD_LEFT_THRESHOLD = float(new_factors[7][0])
    Factors.HEAD_RIGHT_THRESHOLD = float(new_factors[8][0])
    Factors.HEAD_UP_THRESHOLD = float(new_factors[9][0])
    Factors.HEAD_DOWN_THRESHOLD = float(new_factors[10][0])
    f.close()


def configure_factors(frame):
    frame_height, frame_width = frame.shape[:2]
    font_scale = frame_width / 1700

    if len(Outcomes.ON_SCREEN) == 1 and Outcomes.ON_SCREEN[0] != 'face mesh' and Outcomes.ON_SCREEN[0] != 'pose mesh':
        utils.colorBackgroundText(frame, f'Setting {Outcomes.ON_SCREEN[0]} factor...', Factors.FONTS, font_scale * 2,
                                  (round(frame_width / 4), round(frame_height * 2 / 21)), 2, utils.RED, utils.YELLOW)
        if Outcomes.ON_SCREEN[0] == 'lips':
            KEYS[32] = "Factors.SET_THRESHOLD = True"
            utils.colorBackgroundText(frame, "Press 'space' to set threshold...", Factors.FONTS, font_scale,
                                      (round(frame_width / 3), round(frame_height * 20 / 21)), 2, utils.RED,
                                      utils.YELLOW)
            if Factors.SET_THRESHOLD and Outcomes.ON_SCREEN[0] == 'lips':
                Factors.LIPS_RATIO_FACTOR = Outcomes.MOUTH_RATIO[0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD = False
                del KEYS[32]

        elif Outcomes.ON_SCREEN[0] == 'eyes':
            KEYS[32] = "Factors.SET_THRESHOLD = True"
            utils.colorBackgroundText(frame, "Press 'space' to set threshold...", Factors.FONTS, font_scale,
                                      (round(frame_width / 3), round(frame_height * 20 / 21)), 2, utils.RED,
                                      utils.YELLOW)
            if Factors.SET_THRESHOLD and Outcomes.ON_SCREEN[0] == 'eyes':
                Factors.EYES_RATIO_FACTOR = Outcomes.EYES_RATIO[0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD = False
                del KEYS[32]

        elif Outcomes.ON_SCREEN[0] == 'pupils':
            utils.colorBackgroundText(frame, "Press 'r', 'l', 'u' or 'd' to set threshold...", Factors.FONTS,
                                      font_scale,
                                      (round(frame_width / 4), round(frame_height * 20 / 21)), 2, utils.RED,
                                      utils.YELLOW)
            KEYS[108] = "Factors.SET_THRESHOLD_LEFT = True"
            KEYS[114] = "Factors.SET_THRESHOLD_RIGHT = True"
            KEYS[117] = "Factors.SET_THRESHOLD_UP = True"
            KEYS[100] = "Factors.SET_THRESHOLD_DOWN = True"

            if Factors.SET_THRESHOLD_LEFT and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_LEFT_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_LEFT = False
                del KEYS[108]
            if Factors.SET_THRESHOLD_RIGHT and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_RIGHT_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_RIGHT = False
                del KEYS[114]
            if Factors.SET_THRESHOLD_UP and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_UP_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_UP = False
                del KEYS[117]
            if Factors.SET_THRESHOLD_DOWN and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_DOWN_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_DOWN = False
                del KEYS[100]

        elif Outcomes.ON_SCREEN[0] == 'head':
            utils.colorBackgroundText(frame, "Press 'r', 'l', 'u' or 'd' to set threshold...", Factors.FONTS,
                                      font_scale,
                                      (round(frame_width / 4), round(frame_height * 20 / 21)), 2, utils.RED,
                                      utils.YELLOW)
            KEYS[108] = "Factors.SET_THRESHOLD_LEFT = True"
            KEYS[114] = "Factors.SET_THRESHOLD_RIGHT = True"
            KEYS[117] = "Factors.SET_THRESHOLD_UP = True"
            KEYS[100] = "Factors.SET_THRESHOLD_DOWN = True"
            if Factors.SET_THRESHOLD_LEFT and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_LEFT_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_LEFT = False
                del KEYS[108]
            if Factors.SET_THRESHOLD_RIGHT and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_RIGHT_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_RIGHT = False
                del KEYS[114]
            if Factors.SET_THRESHOLD_UP and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_UP_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_UP = False
                del KEYS[117]
            if Factors.SET_THRESHOLD_DOWN and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_DOWN_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_DOWN = False
                del KEYS[100]
        elif Outcomes.ON_SCREEN[0] == 'hands':
            utils.colorBackgroundText(frame, "Press 'space' to set threshold...", Factors.FONTS, font_scale,
                                      (round(frame_width / 3), round(frame_height * 20 / 21)), 2, utils.RED,
                                      utils.YELLOW)
            KEYS[32] = "Factors.SET_THRESHOLD = True"
            if Factors.SET_THRESHOLD and Outcomes.ON_SCREEN[0] == 'hands':
                Factors.HAND_FACE_DISTANCE_FACTOR = min(Outcomes.HANDS_FACE_RATIO[0])
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD = False
                del KEYS[32]
    else:
        utils.colorBackgroundText(frame, f'Choose a configurable part', Factors.FONTS, font_scale * 2,
                                  (round(frame_width / 4), round(frame_height * 2 / 21)), 2, utils.RED, utils.YELLOW)