import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, MaxPooling3D, TimeDistributed, Dropout, \
    CuDNNLSTM
import tensorflow as tf
from collections import Counter
import pandas as pd
import statistics
import random
import sys
import os
from tensorflow.keras.utils import plot_model

past_frames = 60
rows = 100
cols = 200
channels = 1


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.test.is_gpu_available())

balanced_data = list(np.load('.File/preprocessed_data.npy', allow_pickle=True))

train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
    np.asarray([item[0] for item in balanced_data]), np.array([item[1] for item in balanced_data]), test_size=0.2,
    random_state=233)
# Step5 : Define model

model = Sequential()
model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                     recurrent_dropout=0.2, return_sequences=True, input_shape=(past_frames, rows, cols, 1)))
model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
model.add(Flatten())
model.add(Dense(16, activation="softmax"))

plot_model(model, to_file='Files/convlstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step6 : Train model


model.fit(np.asarray(train_inputs), np.array(train_outputs), epochs=1, batch_size=8,
          validation_data=(np.asarray(test_inputs), np.array(test_outputs)))

# Step7 : Test model

model.save('my_model{' + str(past_frames) + 'steps}.h5')
print('Model saved.')
loaded_model = tf.keras.models.load_model('my_model{' + str(past_frames) + 'steps}.h5')
print('Model loaded.')
test_loss, test_acc = loaded_model.evaluate(np.asarray(test_inputs), np.array(test_outputs))
print('Test accuracy:', test_acc)
