

# 手部運動的姿勢座標
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE


def get_labels_of(ax):
    labels = (
        [f'{ax}_right_hand_{i}' for i in range(21)] +
        [f'{ax}_left_hand_{i}' for i in range(21)] +
        [f'{ax}_pose_{i}' for i in POSE]
    )
    return labels


# x, y, z 軸的標籤
X_LABELS = get_labels_of('x')
Y_LABELS = get_labels_of('y')
Z_LABELS = get_labels_of('z')

# 所有軸的標籤，也就是待訓練的特徵
FEATURES = X_LABELS + Y_LABELS + Z_LABELS

# for feat in FEATURES:
#     print(f'{feat:20s}', end = '')