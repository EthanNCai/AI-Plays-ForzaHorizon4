import cv2
import time
from Utilities.grabscreen import grab_screen
import numpy as np
from Utilities.directkeys import PressKey, ReleaseKey, W, A, S, D
import tensorflow as tf

t_time = 0.001
def translate_array(array):
    if np.array_equal(array, [1, 0, 0, 0]):
        return "单纯油门"
    elif np.array_equal(array, [1, 1, 0, 0]):
        return "油门同时左转"
    elif np.array_equal(array, [1, 0, 0, 1]):
        return "油门同时右转"
    elif np.array_equal(array, [0, 0, 1, 0]):
        return "单纯的刹车"
    elif np.array_equal(array, [0, 1, 1, 0]):
        return "刹车同时左转"
    elif np.array_equal(array, [0, 0, 1, 1]):
        return "刹车同时右转"
    elif np.array_equal(array, [0, 1, 0, 0]):
        return "单纯左转"
    elif np.array_equal(array, [0, 0, 0, 1]):
        return "单纯右转"
    else:
        return "其他操作"


def decode_value(encoded_input):
    decoding_dict = {
        0: [0, 0, 0, 0],
        1: [1, 0, 0, 0],
        2: [0, 1, 0, 0],
        3: [0, 0, 1, 0],
        4: [0, 0, 0, 1],
        5: [1, 1, 0, 0],
        6: [1, 0, 1, 0],
        7: [1, 0, 0, 1],
        8: [0, 1, 1, 0],
        9: [0, 1, 0, 1],
        10: [0, 0, 1, 1],
        11: [1, 1, 1, 0],
        12: [1, 1, 0, 1],
        13: [1, 0, 1, 1],
        14: [0, 1, 1, 1],
        15: [1, 1, 1, 1]
    }
    decoded_value = decoding_dict.get(encoded_input, [-1, -1, -1, -1])

    return decoded_value


def press(key_output):
    if key_output[0] == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    if key_output[1] == 1:
        PressKey(A)
    else:
        ReleaseKey(A)
    if key_output[2] == 1:
        PressKey(S)
    else:
        ReleaseKey(S)
    if key_output[3] == 1:
        PressKey(D)
    else:
        ReleaseKey(D)
    time.sleep(t_time)

loaded_model = tf.keras.models.load_model('my_model.h5')
while True:
    # Grab the screen image
    screen = grab_screen(display_index=1, region=(200, 250, 700, 530))

    height, width, _ = screen.shape

    # Calculate midpoints coordinates
    left_mid = (0, height // 2)
    right_mid = (width - 1, height // 2)
    top_mid = (width // 2, 0)
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)

    # Create a mask image
    mask = np.zeros(screen.shape[:2], dtype=np.uint8)
    roi_corners = np.array([[left_mid, top_mid, right_mid, bottom_right, bottom_left]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    # Apply the mask

    # Apply erosion and dilation
    kernel = np.ones((3, 3), np.uint8)
    screen = cv2.erode(screen, kernel, iterations=1)
    screen = cv2.dilate(screen, kernel, iterations=1)

    # Edge detection
    edges = cv2.Canny(screen, 255, 255)
    roi = cv2.bitwise_and(edges, edges, mask=mask)
    # Display the edgesssss

    output = cv2.resize(roi, (200, 100))

    test_inputs = np.expand_dims([output], axis=-1)

    prediction = loaded_model.predict(test_inputs)
    result_literal = translate_array(decode_value(np.argmax(prediction[0])))
    print(result_literal)
    press(decode_value(np.argmax(prediction[0])))
    cv2.imshow('output', output)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
