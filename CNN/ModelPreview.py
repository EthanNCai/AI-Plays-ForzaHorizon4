import numpy as np
import cv2
import tensorflow as tf
import Utilities.onehot

data_path = 'Files/train_data.npy'
model_path = 'Files/cv_cnn_model.h5'


def translate_wasd(lst):
    output = ''
    if lst[0] == 1:
        output += 'W'
    if lst[1] == 1:
        output += 'A'
    if lst[2] == 1:
        output += 'S'
    if lst[3] == 1:
        output += 'D'
    if len(output) == 0:
        output = 'nothing'
    return output


train_data = list(np.load(data_path, allow_pickle=True))
loaded_model = tf.keras.models.load_model(model_path)
for index, data in enumerate(train_data):
    img = data[0]
    target = data[1]
    cv2.imshow('test', img)

    test_inputs = np.expand_dims([img], axis=-1)
    prediction = loaded_model.predict(test_inputs)
    prediction_argmax = np.argmax(prediction[0])
    one_hot_result = [np.eye(prediction[0].shape[0])[prediction_argmax].astype(int).tolist()]

    ground_truth_action = translate_wasd(target)
    predicted_action = translate_wasd(Utilities.onehot.onehot_decode(one_hot_result)[0])

    print('\rframe:', index, 'truth:', ground_truth_action, 'pred:', predicted_action, end='')
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break