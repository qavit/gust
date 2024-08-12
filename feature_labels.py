"""Extract the labels and indices ​​of keypoint coordinates.

Description:
    This program/module contains the following lists.
    - `RPOSE` and `LPOSE` are the lists of landmark numbers related to \
        right arm pose and left arm pose, repsectively.
        - `RPOSE = [14, 16, 18, 20, 22]`
        - `LPOSE = [13, 15, 17, 19, 21]`
    - `X_IDX`, `Y_IDX`, `Z_IDX` are the list of labels related to x, y, z \
        coordinates.
    - `RHAND_IDX` and `LHAND_IDX` are the list of indices related to right \
        hand and left hand, repsectively.
    - `RPOSE_IDX` and `LPOSE_IDX` are the list of indices related to right \
        arm pose and left arm pose, repsectively.

Usage:
    Run the following command in shell to view these lists.
    `python feature_labels.py -V <list_name>`

"""

from auxilaries import glance_list
import argparse

# CONSTANTS
N_kp_hand = 21
RPOSE = [14, 16, 18, 20, 22]  # pose related to right arm
LPOSE = [13, 15, 17, 19, 21]  # pose related to left arm
POSE = LPOSE + RPOSE
# For MediaPipe pose landmarker, see more info at
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker


def get_labels_of(ax: str) -> list:
    """Obtain all hand-related lables of a specific coordinate."""
    labels = (
        [f'{ax}_right_hand_{i}' for i in range(N_kp_hand)] +
        [f'{ax}_left_hand_{i}' for i in range(N_kp_hand)] +
        [f'{ax}_pose_{i}' for i in POSE]
    )
    return labels


# get hand-related labels of x, y, z axes
X_LABELS = get_labels_of('x')
Y_LABELS = get_labels_of('y')
Z_LABELS = get_labels_of('z')

# lables of all axes, i.e., features to be used in deep learning models
FEATURES = X_LABELS + Y_LABELS + Z_LABELS


def extract_idx_by(pattern, labels=FEATURES, pose=None):
    if not pose:
        indices = [i for i, col in enumerate(labels)
                   if pattern in col]
    else:
        indices = [i for i, col in enumerate(labels)
                   if pattern in col and int(col[-2:]) in pose]
    return indices


X_IDX = extract_idx_by("x_")
Y_IDX = extract_idx_by("y_")
Z_IDX = extract_idx_by("z_")
RHAND_IDX = extract_idx_by("right")
LHAND_IDX = extract_idx_by("left")
RPOSE_IDX = extract_idx_by("pose", pose=RPOSE)
LPOSE_IDX = extract_idx_by("pose", pose=LPOSE)

# dictrionary for argparse
parse_dict = {
    'X_LABELS': X_LABELS,
    'Y_LABELS': Y_LABELS,
    'Z_LABELS': Z_LABELS,
    'FEATURES': FEATURES,
    'X_IDX': X_IDX,
    'Y_IDX': Y_IDX,
    'Z_IDX': Z_IDX,
    'RHAND_IDX': RHAND_IDX,
    'LHAND_IDX': LHAND_IDX,
    'RPOSE_IDX': RPOSE_IDX,
    'LPOSE_IDX': LPOSE_IDX,
}

label_set = ['X_LABELS', 'Y_LABELS', 'Z_LABELS', 'FEATURES']
idx_set = ['RHAND_IDX', 'LHAND_IDX', 'RPOSE_IDX', 'LPOSE_IDX']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print ')
    parser.add_argument('-v', type=str,
                        help='Print the variables verbosely.')
    parser.add_argument('-b', type=str,
                        help='Print the variables briefly.')
    args = parser.parse_args()

    if args.v is not None:
        for i in parse_dict[args.v]:
            ending = '\n' if args.v in label_set else ', '
            print(i, end=ending)

    if args.b is not None:
        print(glance_list(parse_dict[args.b]))
