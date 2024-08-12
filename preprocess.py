import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import argparse


from settings import FRAME_LEN, PREPROC_DIR
from feature_labels import FEATURES
from feature_labels import RHAND_IDX, LHAND_IDX, RPOSE_IDX, LPOSE_IDX
from characters import char_to_num
from characters import START_TOKEN, END_TOKEN, PAD_TOKEN_IDX


def resize_pad(x, frame_len=FRAME_LEN):
    """Adjust the size of the input tensor and apply padding if necessary.

    If the number of frames (first dimension) is less than `frame_len`,
    the tensor is padded to ensure it has `frame_len` frames.
    If the number of frames is equal to or greater than `frame_len`,
    the tensor is resized to ensure it has exactly `frame_len` frames.

    Args:
        x (tf.Tensor): Input tensor with shape (num_frames, height, width).
        frame_len (int): length of frame. Defaults to FRAME_LEN (128).

    Returns:
        x (tf.Tensor): Tensor with adjusted size and padding applied if needed.
    """
    if tf.shape(x)[0] < frame_len:
        x = tf.pad(x, ([[0, frame_len - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (frame_len, tf.shape(x)[1]))
    return x


def pre_process(x):
    """Process input data to determine the dominant hand based on NaN values
    and normalize hand and pose data.

    The function determines the dominant hand (the one with fewer NaN values)
    and processes the hand and pose data accordingly.
    Hand data and pose data are normalized by calculating their mean and
    standard deviation.

    Args:
        x (tf.Tensor): Input tensor with shape (num_frames, num_features).

    Returns:
        x (tf.Tensor): Processed tensor with normalized and padded data.
    """

    # Extract landmark tensors according to the indices.
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)

    # Check whether there are NaN values ​​in the right and left hand data.
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)

    # Count the number of frames containing NaN values ​​in the right and left
    # hands.
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)

    def split_xyz(body_part, idx):
        """Split the x, y, z data for the body part"""
        part_x = body_part[:, 0*(len(idx)//3): 1*(len(idx)//3)]
        part_y = body_part[:, 1*(len(idx)//3): 2*(len(idx)//3)]
        part_z = body_part[:, 2*(len(idx)//3): 3*(len(idx)//3)]
        return part_x, part_y, part_z

    def mirror(body_part, idx):
        """Perform the left-right symmetry transformation,
        i.e. `x` -> `1-x`."""
        part_x, part_y, part_z = split_xyz(body_part, idx)
        return tf.concat([1 - part_x[0], part_y[1], part_z[2]], axis=1)

    def unknown_action(body_part, idx):
        part_x, part_y, part_z = split_xyz(body_part, idx)
        return tf.concat([part_x[..., tf.newaxis],
                          part_y[..., tf.newaxis],
                          part_z[..., tf.newaxis]],
                         axis=-1)

    if rnans > lnans:
        dom_hand = mirror(lhand, LHAND_IDX)
        dom_pose = mirror(lpose, LPOSE_IDX)
    else:
        dom_hand = rhand
        dom_pose = rpose

    dom_hand = unknown_action(dom_hand, LHAND_IDX)

    mean = tf.math.reduce_mean(dom_hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(dom_hand, axis=1)[:, tf.newaxis, :]
    dom_hand = (dom_hand - mean) / std

    dom_pose = unknown_action(dom_pose, LPOSE_IDX)

    x = tf.concat([dom_hand, dom_pose], axis=1)
    x = resize_pad(x)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x


def decode_fn(record_bytes: bytes):
    # Define the parsing schema
    # 1. Take feature labels in FEATURES as variable-length features
    schema = {col: tf.io.VarLenFeature(dtype=tf.float32) for col in FEATURES}
    # 2. Take 'phrase' as fixed-length features
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)

    # Parse serialized data
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]

    # Reconstruct landmarks:
    # Convert each feature from sparse tensor to dense tensor
    landmarks = [tf.sparse.to_dense(features[col]) for col in FEATURES]

    # Transpose to preserve original data shape
    landmarks = tf.transpose(landmarks)

    return landmarks, phrase


hash_table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=list(char_to_num.keys()),
                    values=list(char_to_num.values()),
                    ),
                default_value=tf.constant(-1),
                name="class_weight"
                )


def convert_fn(landmarks, phrase):
    # Add start and end pointers to phrase.
    phrase = START_TOKEN + phrase + END_TOKEN
    phrase = tf.strings.bytes_split(phrase)
    phrase = hash_table.lookup(phrase)

    # Vectorize and add padding.
    phrase = tf.pad(phrase,
                    paddings=[[0, 64 - tf.shape(phrase)[0]]],
                    mode='CONSTANT',
                    constant_values=PAD_TOKEN_IDX)

    # Apply pre_process function to the landmarks.
    return pre_process(landmarks), phrase


def preprocess(tf_records: str, batch_size: int = 64):
    X = tf.data.TFRecordDataset(tf_records[:train_len])
    print(type(X))
    X = X.map(decode_fn)
    print(type(X))
    #X = X.map(convert_fn)
    # print(type(X))
    # X = X.batch(batch_size)
    # print(type(X))
    # X = X.prefetch(buffer_size=tf.data.AUTOTUNE)
    # print(type(X))
    # X = X.cache()
    # print(type(X))
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse the TFRecord.')
    parser.add_argument('--file', '-f', type=str, default='105143404.tfrecord',
                        help='The path to the TFRecord file.')
    args = parser.parse_args()

    print(args.file)

    train_len = int(0.8 * len(args.file))

    train_ds = preprocess(args.file[:train_len])
    #valid_ds = preprocess(args.file[train_len:])
