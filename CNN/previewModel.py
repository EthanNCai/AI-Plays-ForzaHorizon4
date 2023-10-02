import numpy as np
import cv2

import tensorflow as tf


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


file_name = '../train_data.npy'
train_data = list(np.load(file_name, allow_pickle=True))
loaded_model = tf.keras.models.load_model('my_model.h5')
for data in train_data:
    img = data[0]
    target = data[1]
    cv2.imshow('test', img)
    result = translate_array(target)
    test_inputs = np.expand_dims([img], axis=-1)
    prediction = loaded_model.predict(test_inputs)
    result_literal = translate_array(decode_value(np.argmax(prediction[0])))
    print(result_literal)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
