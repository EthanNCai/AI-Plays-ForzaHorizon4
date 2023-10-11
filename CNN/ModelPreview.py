import numpy as np
import cv2
import tensorflow as tf
import Utilities.onehot as oh
import Utilities.translate_result as tr

data_path = 'Files/train_data.npy'
model_path = 'Files/cv_cnn_model.h5'

# load data and models
train_data = list(np.load(data_path, allow_pickle=True))
loaded_model = tf.keras.models.load_model(model_path)

# for each frames in dataset...
for index, data in enumerate(train_data):
    img = data[0]
    target = data[1]

    # display the image
    cv2.imshow('test', img)
    # preprocess the image
    test_inputs = np.expand_dims([img], axis=-1)

    # generate prediction by model
    prediction = loaded_model.predict(test_inputs)
    # acquire the argmax
    prediction_argmax = np.argmax(prediction[0])
    # generate a one_hot encoded result according to the predicted argmax value
    one_hot_result = [np.eye(prediction[0].shape[0])[prediction_argmax].astype(int).tolist()]
    # generate the prediction confidence
    confidence = max(prediction[0])
    # interpret the keyboard action
    ground_truth_action = tr.translate_wasd(target)
    predicted_action = tr.translate_wasd(oh.onehot_decode(one_hot_result)[0])

    # print
    print('\rframe:', index,
          'truth:', ground_truth_action,
          'pred:', predicted_action,
          'confidence:', confidence,
          end='')

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
