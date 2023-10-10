import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Flatten, Dense, MaxPooling3D
import tensorflow as tf
import os
from tensorflow.keras.utils import plot_model

past_frames = 20
rows = 100
cols = 200
channels = 1

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print('Are we using GPU?:', tf.test.is_gpu_available())

balanced_data = np.load('Files/preprocessed_data.npy', allow_pickle=True)
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
    np.asarray([item[0] for item in balanced_data]), np.array([item[1] for item in balanced_data]),
    test_size=0.2,
    random_state=233)

# Step2 : Define model

model = Sequential()
model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                     recurrent_dropout=0.2, return_sequences=True, input_shape=(train_inputs.shape[1],
                                                                                train_inputs.shape[2],
                                                                                train_inputs.shape[3],
                                                                                train_inputs.shape[4])))
model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
model.add(Flatten())
model.add(Dense(16, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plot_model(model, to_file='Files/convlstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)

# Step3 : Train model

print('input  shape:', train_inputs.shape)
print('output shape:', train_outputs.shape)

model.fit(train_inputs, train_outputs, epochs=1, batch_size=4,
          validation_data=(test_inputs, test_outputs))

# Step4 : Evaluate model

model.save('cv_convlstm_model{' + str(past_frames) + 'steps}.h5')
print('Model saved.')
loaded_model = tf.keras.models.load_model('cv_convlstm_model{' + str(past_frames) + 'steps}.h5')
print('Model loaded.')
test_loss, test_acc = loaded_model.evaluate(np.asarray(test_inputs), np.array(test_outputs))
print('Val. accuracy:', test_acc)
