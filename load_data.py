import argparse
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import matlotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from matplotlib import rcParams

from settings import TRAIN_META, TRAIN_DIR

# ============================================================

metadata_df = pd.read_csv(TRAIN_META)

# ============================================================


# Extracts landmark data for both hands using the MediaPipe library
def extract_landmarks(keypoint: pd.Series, part: str) -> tuple:
    """Extracts and visualizes the landmarks for a specified body part from \
        the given keypoints.

    Parameters:
        keypoint (pd.Series): A Pandas Series containing the x, y, z \
            coordinates of various keypoints.
        part (str): The body part for which landmarks should be extracted \
            (e.g., 'right_hand', 'left_hand').

    Returns:
        tuple: A tuple containing the image with landmarks drawn and the list \
            of landmarks.
    """
    # Load MediaPipe's pose and hand detection models, along with drawing
    # utilities
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    default_style = mp_styles.get_default_hand_landmarks_style()

    # Retrieve the x, y, z coordinates for the specified body part
    xs = keypoint.filter(regex=f"x_{part}.*").values
    ys = keypoint.filter(regex=f"y_{part}.*").values
    zs = keypoint.filter(regex=f"z_{part}.*").values

    # Initialize a blank image and an empty list of landmarks
    image = np.zeros((600, 600, 3))
    landmarks = landmark_pb2.NormalizedLandmarkList()

    # Add landmark data to the list
    for x, y, z in zip(xs, ys, zs):
        landmarks.landmark.add(x=x, y=y, z=z)

    # Draw landmarks on the image
    mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=default_style)

    return image, landmarks


def get_both_hands(seq_df: pd.DataFrame) -> tuple:
    """Extracts and visualizes landmarks for both hands across a sequence of\
        data points.

    Parameters:
        seq_df (pd.DataFrame): A DataFrame where each row contains the \
            keypoints for a frame in the sequence.

    Returns:
        tuple: A tuple containing a list of images with landmarks drawn and a \
            list of landmarks for both hands.
    """
    images = []
    landmarks = []

    # Iterate through each row in the sequence
    for seq_idx in range(len(seq_df)):
        keypoints = seq_df.iloc[seq_idx]

        # Extract landmark data for the right hand and left hand
        results_R = extract_landmarks(keypoints, 'right_hand')
        results_L = extract_landmarks(keypoints, 'left_hand')

        # Store the images and landmark data
        images.append([results_R[0].astype(np.uint8),
                       results_L[0].astype(np.uint8)])

        landmarks.append([results_R[1], results_L[1]])

    return images, landmarks


def create_animation(images):
    """Creates an animation from a sequence of images.

    Parameters:
        images (list): A list of images to be used in the animation.

    Returns:
        anim (matplotlib.animation.FuncAnimation): The created animation \
            object.
    """
    rcParams['animation.embed_limit'] = 2**128
    rcParams['savefig.pad_inches'] = 0
    rc('animation', html='jshtml')

    # Create a figure with a size of 6x9 inches
    fig = plt.figure(figsize=(6, 9))

    # Add an Axes object that fills the entire figure
    # and hides both the x-axis and y-axis
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Display the first image using a grayscale color map
    im = ax.imshow(images[0], cmap="gray")

    # Close the figure to allow the animation to run in the background
    plt.close(fig)

    # Animation function that updates the image based on the
    # current frame index
    def animate_func(frame_idx: int):
        im.set_array(images[frame_idx])
        return [im]

    # Create the animation with a frame interval of 100 ms (10 frames per
    # second)
    anim = animation.FuncAnimation(fig, animate_func,
                                   frames=len(images),
                                   interval=1000/10)

    return anim


class SequenceSample():
    def __init__(self, sample_index):
        sequence_df = metadata_df.iloc[sample_index]
        self.name = f'Sequence {sample_index:04d}'
        self.sequence_id = sequence_df['sequence_id']
        self.file_id = sequence_df['file_id']
        self.phrase = sequence_df['phrase']

    def get_path(self):
        return os.path.join(TRAIN_DIR, f'{self.file_id}.parquet')

    def get_df(self):
        pq_source = self.get_path()
        pq_filter = [[('sequence_id', '=', self.sequence_id)],]
        pq_table = pq.read_table(pq_source, filters=pq_filter)
        return pq_table.to_pandas()


if __name__ == '__name__':
    print(metadata_df)
    # parser = argparse.ArgumentParser(description='View the TFRecord.')
    # parser.add_argument('--file', '-f', type=str, default='105143404.tfrecord',
    #                     help='The path to the TFRecord file.')
    # parser.add_argument('--info', '-i', action='store_true',
    #                     help='Show info.')
    # args = parser.parse_args()

    # sample_sequence = SequenceSample(0)

    # if args.info:
    #     print(sample_sequence.__dict__)
