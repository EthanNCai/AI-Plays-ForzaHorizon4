import numpy as np
import cv2
import queue
import tensorflow as tf
import Utilities.onehot as oh
import Utilities.translate_result as tr
from Utilities.grabscreen import grab_screen
from Utilities.cv_crop_processing import crop_screen
import time
import os


past_frames = 20
file_name = 'Files/train_data.npy'

time_checkpoint_a, time_checkpoint_b = 0, 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 1
frames_counter = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('Are we using GPU?:', tf.test.is_gpu_available())

train_data = list(np.load(file_name, allow_pickle=True))
loaded_model = tf.keras.models.load_model('Files/cv_convlstm_model{20steps}.h5')
q = queue.Queue(maxsize=past_frames)

while True:
    frames_counter += 1

    # for fps calculation
    time_checkpoint_a = time.time()
    # acquire screen signal.
    # then, cropping, gray scaling, post-processing
    screen = grab_screen(display_index=1, region=(0, 0, 1280, 720))
    screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    cropped_screen = crop_screen(screen, trim_rate=0.2)
    cropped_gray = cv2.cvtColor(cropped_screen, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(cropped_gray, (200, 100))

    print(resized_image)

    # manipulate the gray scale image matrix shape (width,height) -> (width,height,depth)
    test_inputs = np.expand_dims(screen, axis=-1)

    # this is the past-n-frames queue for generating sequential data for LSTM network
    if q.qsize() == past_frames:
        q.get()
    q.put(test_inputs)
    if q.qsize() < past_frames:
        continue

    sequential_input = np.asarray(list(q.queue))
    # data feed to the model has to be inside a list, so I did this.
    sequential_input = np.expand_dims(sequential_input, axis=0)
    prediction = loaded_model.predict(sequential_input)

    # acquire the argmax
    prediction_argmax = np.argmax(prediction[0])
    # generate a one_hot encoded result according to the predicted argmax value
    one_hot_result = [np.eye(prediction[0].shape[0])[prediction_argmax].astype(int).tolist()]
    # generate the prediction confidence
    confidence = max(prediction[0])
    # interpret the keyboard action
    predicted_action = tr.translate_wasd(oh.onehot_decode(one_hot_result)[0])

    # for fps calculation
    time_checkpoint_b = time.time()
    # insert the fps information to the screen
    fps = 1 / (time_checkpoint_b - time_checkpoint_a)
    cv2.putText(screen_rgb, str("{:.2f}".format(fps)), (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # print some real-time information
    cv2.imshow('test', screen_rgb)
    print('\rframe:', frames_counter,
          'pred:', predicted_action,
          'confidence:', confidence,
          'fps:', fps,
          end='')

    # some cv2 ritual
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
