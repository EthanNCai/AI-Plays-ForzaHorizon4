import cv2
import numpy as np


def edge_processing(screen_in, resize_width=200, resize_height=100):

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
    screen_out = cv2.resize(roi, (resize_width, resize_height))
    return screen_out
