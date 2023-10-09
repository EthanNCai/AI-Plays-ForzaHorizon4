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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.test.is_gpu_available())


def one_hot_encode(number, num_classes):
    encoding_tem = [0] * num_classes  # 创建一个全零列表
    encoding_tem[number] = 1  # 将指定位置设置为 1
    return encoding_tem


def encode_list(lst):
    encoding_dict = {
        (0, 0, 0, 0): 0,
        (1, 0, 0, 0): 1,
        (0, 1, 0, 0): 2,
        (0, 0, 1, 0): 3,
        (0, 0, 0, 1): 4,
        (1, 1, 0, 0): 5,
        (1, 0, 1, 0): 6,
        (1, 0, 0, 1): 7,
        (0, 1, 1, 0): 8,
        (0, 1, 0, 1): 9,
        (0, 0, 1, 1): 10,
        (1, 1, 1, 0): 11,
        (1, 1, 0, 1): 12,
        (1, 0, 1, 1): 13,
        (0, 1, 1, 1): 14,
        (1, 1, 1, 1): 15
    }
    encoded_lst = []
    for sub_lst in lst:
        encoded_value = encoding_dict.get(tuple(sub_lst), 0)
        encoded_lst.append(encoded_value)
    return encoded_lst


rows = 100
cols = 200
channels = 1
past_frames = 60

# Step1 : load datasets
data = np.load('Files/train_data.npy', allow_pickle=True)

inputs = np.asarray([item[0] for item in data])
outputs = np.array([item[1] for item in data])
inputs = np.expand_dims(inputs, axis=-1)

# Step2 : encode datasets

outputs = np.array(encode_list(outputs))
real_outputs = []
for item in outputs:
    encoding = one_hot_encode(item, 16)
    real_outputs.append(encoding)
outputs = np.array(real_outputs)

# Step3 : Generate sequential data for ConvLSTM

X, y = list(), list()
for i in range(len(inputs)):
    end_ix = i + past_frames
    if end_ix > len(inputs) - 1:
        break
    seq_x, seq_y = inputs[i:end_ix], outputs[end_ix]
    X.append(seq_x)
    y.append(seq_y)

unbalanced_data = [[a, b] for a, b in zip(X, y)]
counter = Counter(map(tuple, pd.DataFrame(unbalanced_data)[1]))
counter_list = [(label, count) for label, count in counter.items()]
balancing_target = int(statistics.mean(count for _, count in counter_list))

print('original_data:')
print(counter)

# manipulate data according to balancing target
balanced_data = []
for label, count in counter_list:
    label_data = [[i, l] for i, l in unbalanced_data if (label == l).all()]
    if count < balancing_target:
        balanced_data += label_data
    else:
        balanced_data += random.sample(label_data, balancing_target)

# shuffle data
random.shuffle(balanced_data)

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
