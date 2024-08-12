
import os
import json
import tensorflow as tf
from auxilaries import *

FRAME_LEN = 128
CHAR_TO_NUM = os.path.join(ROOT, 'character_to_prediction_index.json')
PREPROC_DIR = '/kaggle/working/preprocessed'

with open(CHAR_TO_NUM, "r") as f:
    char_to_num = json.load(f)

pad_token_idx = 59
start_token_idx = 60
end_token_idx = 61

char_to_num['P'] = pad_token_idx  # pad token
char_to_num['<'] = start_token_idx  # start token
char_to_num['>'] = end_token_idx  # end token

num_to_char = {j: i for i, j in char_to_num.items()}


# 此函式用來調整影格的大小並添加填充（pad）。
def resize_pad(x):
    # 檢查張量 'x' 的第一維度（通常是影格數）是否小於 FRAME_LEN。
    if tf.shape(x)[0] < FRAME_LEN:
        # 如果影格數小於 FRAME_LEN，則對 'x' 進行填充，填充的大小是 (FRAME_LEN - 當前影格數)，
        # 只對第一維度進行填充，其餘維度不變。
        x = tf.pad(x, ([[0, FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        # 如果影格數等於或大於 FRAME_LEN，則調整 'x' 的大小以確保它的影格數為 FRAME_LEN。
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x

# 根據 NaN 值的數量檢測主導手。
# 主導手的 NaN 值較少，因為它在影格中移動。
def pre_process(x):
    # 根據索引從數據中提取右手、左手、右手臂姿勢和左手臂姿勢的資料。
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)
    
    # 檢查右手和左手資料中是否有 NaN 值。
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)
    
    # 計算右手和左手中含有 NaN 值的影格數。
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)
    
    # 確定主導手（NaN 值較少的手為主導手）。
    if rnans > lnans:
        # 如果右手的 NaN 值較多，則選擇左手為主導手。
        hand = lhand
        pose = lpose
        
        # 將手部的 x、y、z 資料拆分並進行反向處理（例如對 x 分量取 1-x）。
        hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
        hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
        hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
        hand = tf.concat([1-hand_x, hand_y, hand_z], axis=1)
        
        # 將手臂姿勢的 x、y、z 資料拆分並進行反向處理（例如對 x 分量取 1-x）。
        pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
        pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
        pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
        pose = tf.concat([1-pose_x, pose_y, pose_z], axis=1)
    else:
        # 如果右手的 NaN 值較少，則選擇右手為主導手。
        hand = rhand
        pose = rpose
    
    # 將手部資料拆分為 x、y、z 分量，並擴展維度以便進行標準化。
    hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
    hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
    hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
    hand = tf.concat([hand_x[..., tf.newaxis], 
                      hand_y[..., tf.newaxis], 
                      hand_z[..., tf.newaxis]], 
                     axis=-1)
    
    # 計算手部數據的均值和標準差，並進行標準化處理。
    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std

    # 將姿勢資料拆分為 x、y、z 分量，並擴展維度以便進行標準化。
    pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
    pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
    pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
    pose = tf.concat([pose_x[..., tf.newaxis], 
                      pose_y[..., tf.newaxis], 
                      pose_z[..., tf.newaxis]], 
                     axis=-1)
    
    # 合併手部和姿勢資料，並進行調整和填充處理。
    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)
    
    # 將 NaN 值替換為 0，並調整張量的形狀。
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x