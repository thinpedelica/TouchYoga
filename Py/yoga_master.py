#!/usr/bin/env python
# coding: UTF-8

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
import math
import numpy as np
import os

from osc_io import OscIO

MODEL_PATH = "../Model/yoga_model.h5"
BASE_LENGTH_HEAD_TO_THROAT = 66.0
IMG_ROW       = 16
IMG_COL       = 16
POINT_NUM     = 14
POSE_NUM      = 12
ARROWABLE_ERROR_LENGTH = 30.0

HOST = '127.0.0.1'
PORT = 8080

out_file_index = 0
out_file_path = "C:\openpose-windows\\windows_project\\tools\\out"

class Point():
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

"""
    x,yのリストをもとに、Point型のリストを作成する
    in: skelton_pos_list
    return: point_list
"""
def create_point_list(skelton_pos_list):
    point_list = []
    for i in range(0, POINT_NUM):
        x_index = i * 2
        y_index = x_index + 1
        point_list.append(Point(skelton_pos_list[x_index], skelton_pos_list[y_index]))

    return point_list

"""
    頭の位置を原点として、x,yの位置を移動する
    in: point_list
    return: converted_point_list
"""
def convert_position(point_list):
    # 原点を頭の位置にする
    is_head = True
    head_x = 0.0
    head_y = 0.0
    converted_point_list = []
    for point in point_list:
        if is_head == True:
            head_x = point.x
            head_y = point.y
            converted_point = Point(0.0, 0.0)
            converted_point_list.append(converted_point)
            is_head = False
        else:
            x = point.x - head_x
            y = point.y - head_y
            converted_point = Point(x, y)
            converted_point_list.append(converted_point)

    return converted_point_list

"""
    頭から喉までの距離を基準に、リサイズ比率を計算する
    in: point_list(原点は頭とする)
    return: resize_ratio
"""
def calc_resize_ratio(point_list):
    length = math.sqrt((point_list[1].x ** 2) + (point_list[1].y ** 2))
    return BASE_LENGTH_HEAD_TO_THROAT / length

"""
    頭から喉までの距離を基準に、スケルトンをリサイズする
    in: point_list
    return: point_list
"""
def resize(point_list):
    resized_point_list = []
    resize_ratio = calc_resize_ratio(point_list)
    for point in point_list:
        x = point.x * resize_ratio
        y = point.y * resize_ratio
        resized_point_list.append(Point(x, y))

    return resized_point_list

"""
    スケルトンの位置リスト(ノーマライズ済み)をNumpy配列に設定する
    in     : point_list
    return : x_test
"""
def set_nparray(point_list):
    x_test = np.zeros((1, IMG_ROW, IMG_COL))
    for point in range(0, POINT_NUM):
        x_test[0, point, point]   = point_list[point].x
        x_test[0, point, point+1] = point_list[point].y

    x_test = x_test.reshape(x_test.shape[0], IMG_ROW, IMG_COL, 1)
    x_test = x_test.astype('float32')

    return x_test

"""
    スケルトンの位置リスト(ノーマライズ済み)を比較し、一致している確率を返却する
    in     : point_list
    return : pose_matching_probabirity
"""
def culc_pose_matching(point_list_left, point_list_right):
    matching_points = 0.0
    # 各ポイントの位置を比較し、誤差の範囲内か否かを判定する
    for point in range(0, POINT_NUM):
        x_left  = point_list_left[0, point, point]
        y_left  = point_list_left[0, point, point+1]
        x_right = point_list_right[0, point, point]
        y_right = point_list_right[0, point, point+1]

        x_error = abs(x_left - x_right)
        y_error = abs(y_left - y_right)

        if (x_error < ARROWABLE_ERROR_LENGTH) and (y_error < ARROWABLE_ERROR_LENGTH):
            matching_points += 1.0

    return matching_points / POINT_NUM

"""
    スケルトンの位置リストから、ノーマライズしたデータセットを作成する
    in     : skelton_pos_list
    return : x_test
"""
def convert_dataset(skelton_pos_list):
    point_list     = create_point_list(skelton_pos_list)
    converted_list = convert_position(point_list)
    resized_list   = resize(converted_list)
    x_test         = set_nparray(resized_list)

    # ToDo Debug
    # write_list(point_list)
    return x_test

"""
    スケルトンのx座標とy座標をファイルに書き込む
    in: point_list
"""
def write_list(point_list):
    global out_file_index
    file_name = out_file_path + os.sep + str(out_file_index) + ".txt"
    try:
        out_file = open(file_name, 'w')
        for point in point_list:
            out_file.write(str(point.x) + " " + str(point.y) + "\n")
    except:
        print("fail to write file.")
    finally:
        out_file.close()
        out_file_index += 1

"""
    スケルトンの位置リストからヨガポーズを推定し、ポーズID, 確率, 前回ポーズを返却する
    in     : model
    in     : skelton_pos_list
    return : pose_id
    return : proba
    return : x_test
"""
def predict_pose(model, skelton_pos_list):
    x_test = convert_dataset(skelton_pos_list)

    proba_array = model.predict_proba(x_test, verbose=0)

    pose_id = proba_array[0].argmax()
    proba   = proba_array[0].max()

    return pose_id, proba, x_test


"""
    クライアントから受信したskeltonに対応するposeのidと確率と前回ポーズ一致確率を返信する
    in : model
    in : osc
"""
def answer_pose(model, osc):
    # (x,y) * 14pos
    while True:
        ret, skelton_pos_list = osc.recv()
        if not ret:
            break
        if len(skelton_pos_list) == 0:
            continue

        pose_id, proba, current_point_array = predict_pose(model, skelton_pos_list)
        print("{}:{}".format(pose_id, proba))

        # v.item() : for numpy type to native python type
        osc.send(pose_id.item(), proba.item())

    osc.close()

"""
    メイン関数
"""
if __name__ == '__main__':
    # モデルの読み込み
    model = load_model(MODEL_PATH)

    # 受信したskeltonに対するポーズの返却
    osc = OscIO()
    answer_pose(model, osc)
