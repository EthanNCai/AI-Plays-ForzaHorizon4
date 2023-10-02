import os.path

import cv2
import time
from Utilities.grabscreen import grab_screen
import numpy as np
from Utilities.getkeys import key_check


def processing_image(screen_in):
    height, width, _ = screen_in.shape

    left_mid = (0, height // 2)
    right_mid = (width - 1, height // 2)
    top_mid = (width // 2, 0)
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)

    mask = np.zeros(screen_in.shape[:2], dtype=np.uint8)
    roi_corners = np.array([[left_mid, top_mid, right_mid, bottom_right, bottom_left]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    kernel = np.ones((3, 3), np.uint8)
    screen_in = cv2.erode(screen_in, kernel, iterations=1)
    screen_in = cv2.dilate(screen_in, kernel, iterations=1)

    edges = cv2.Canny(screen_in, 255, 255)
    roi = cv2.bitwise_and(edges, edges, mask=mask)

    screen_out = cv2.resize(roi, (200, 100))

    return screen_out


def keys_to_output(keys_in):
    # [W,A,S,D]
    # [1,0,0,0] 单纯油门 +
    # [1,1,0,0] 油门同时左转 +
    # [1,0,0,1] 油门同时右转 +
    # [0,0,0,0] 空载
    key_to_output = np.array([0, 0, 0, 0], dtype=np.uint8)

    if 'W' in keys_in:
        key_to_output[0] = 1
    if 'A' in keys_in:
        key_to_output[1] = 1
    if 'S' in keys_in:
        key_to_output[2] = 1
    if 'D' in keys_in:
        key_to_output[3] = 1

    return key_to_output

# code begins here


file_name = '../train_data.npy'
if os.path.isfile(file_name):
    print('模型文件已经存在，将会Append.....')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('模型文件不存在，这是一个全新的开始！')
    training_data = []

for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)

while True:

    screen = grab_screen(display_index=1, region=(200, 250, 700, 530))
    screen_output = processing_image(screen)
    keys = key_check()
    key_output = keys_to_output(keys)

    training_data.append([screen_output, key_output])

    cv2.imshow('output', screen_output)

    if len(training_data) % 500 == 0:
        print(len(training_data))

        # np.array(training_data, dtype=object)

        np.save(file_name, np.array(training_data, dtype=object))
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
