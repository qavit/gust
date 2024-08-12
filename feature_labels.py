# PARAMETERS
N_kp_hand = 21
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

# See more info at
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

if __name__ == "__main__":
    for feat in FEATURES:
        print(f'{feat:20s}', end='')
