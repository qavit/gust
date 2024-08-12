from os.path import join

ROOT = 'data'

CHAR_TO_NUM_JSON = join(ROOT, 'character_to_prediction_index.json')
TRAIN_META = join(ROOT, 'train.csv')
TRAIN_DIR = join(ROOT, 'train_landmarks')
SUPPL_DIR = join(ROOT, 'supplemental_landmarks')
PREPROC_DIR = 'preprocessed'

FRAME_LEN = 128