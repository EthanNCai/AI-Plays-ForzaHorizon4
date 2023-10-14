import numpy as np
import cv2
import queue
import tensorflow as tf
import Utilities.onehot as oh
import Utilities.translate_result as tr
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('Are we using GPU?:', tf.test.is_gpu_available())
past_frames = 20
q = queue.Queue(maxsize=past_frames)
file_name = 'Files/train_data.npy'
train_data = list(np.load(file_name, allow_pickle=True))
loaded_model = tf.keras.models.load_model('Files/cv_convlstm_model{20steps}.h5')

time_checkpoint_a = 0
time_checkpoint_b = 0

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
thickness = 1

for index, data in enumerate(train_data):
    time_checkpoint_a = time.time()

    img = data[0]
    target = data[1]

    test_inputs = np.expand_dims(img, axis=-1)

    if q.qsize() == past_frames:
        q.get()
    q.put(test_inputs)
    if q.qsize() < past_frames:
        continue

    sequential_input = np.asarray(list(q.queue))
    sequential_input = np.expand_dims(sequential_input, axis=0)
    prediction = loaded_model.predict(sequential_input)

    # acquire the argmax
    prediction_argmax = np.argmax(prediction[0])
    # generate a one_hot encoded result according to the predicted argmax value
    one_hot_result = [np.eye(prediction[0].shape[0])[prediction_argmax].astype(int).tolist()]
    # generate the prediction confidence
    confidence = max(prediction[0])
    # interpret the keyboard action
    ground_truth_action = tr.translate_wasd(target)
    predicted_action = tr.translate_wasd(oh.onehot_decode(one_hot_result)[0])

    time_checkpoint_b = time.time()
    fps = 1 / (time_checkpoint_b - time_checkpoint_a)

    cv2.putText(img, str("{:.2f}".format(fps)), (0, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.imshow('test', img)
    print('\rframe:', index,
          'truth:', ground_truth_action,
          'pred:', predicted_action,
          'confidence:', confidence,
          'fps:', fps,
          end='')

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
