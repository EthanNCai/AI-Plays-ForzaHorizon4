import cv2
from Utilities.grabscreen import grab_screen
import numpy as np
import Utilities.edge_processing as ep
import tensorflow as tf
import Utilities.onehot as oh
import Utilities.translate_result as tr
import Utilities.keypress as kp


# load model
loaded_model = tf.keras.models.load_model('Files/cv_cnn_model.h5')

while True:

    # grab screen information, apply some mask according to our model
    # (in this case, edge detection)
    screen = grab_screen(display_index=1, region=(200, 250, 700, 530))
    output = ep.edge_processing(screen)
    test_inputs = np.expand_dims([output], axis=-1)

    # generate prediction by model
    prediction = loaded_model.predict(test_inputs)
    # acquire the argmax
    prediction_argmax = np.argmax(prediction[0])
    # generate a one_hot encoded result according to the prediction argmax value
    one_hot_result = [np.eye(prediction[0].shape[0])[prediction_argmax].astype(int).tolist()]
    # generate the prediction confidence
    confidence = max(prediction[0])
    # interpret the keyboard action

    wasd = oh.onehot_decode(one_hot_result)[0]
    predicted_action = tr.translate_wasd(wasd)

    # do the corresponding keyboard action
    if confidence > 0.4:
        kp.key_press(wasd, confidence)

    # print the current frame info
    print('\rpred:', predicted_action,
          'confidence:', confidence,
          end='')

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
