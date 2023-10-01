import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, MaxPooling3D, TimeDistributed, Dropout, CuDNNLSTM
import tensorflow as tf


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.utils import plot_model

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
past_frames = 10

# Step1 : load datasets
data = np.load('train_data.npy', allow_pickle=True)

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

# Step3 : Generate sequential data for LSTM

X, y = list(), list()
for i in range(len(inputs)):
    # find the end of this pattern
    end_ix = i + past_frames
    # check if we are beyond the sequence
    if end_ix > len(inputs) - 1:
        break
    # gather input and output parts of the pattern
    seq_x, seq_y = inputs[i:end_ix], outputs[end_ix]
    X.append(seq_x)
    y.append(seq_y)

# Step4 : Split data

train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(X, y, test_size=0.3,
                                                                          random_state=42)
# Step5 : Define model

model = Sequential()

model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                     recurrent_dropout=0.2, return_sequences=True, input_shape=(10, 100, 200, 1)))
model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
# model.add(TimeDistributed(Dropout(0.2)))

model.add(Flatten())

model.add(Dense(16, activation="softmax"))
# plot_model(model, to_file='convlstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step6 : Train model

print(np.shape(train_inputs), np.shape(train_outputs))

model.fit(np.asarray(train_inputs), np.array(train_outputs), epochs=1, batch_size=4,
          validation_data=(np.asarray(test_inputs), np.array(test_outputs)))

# Step7 : Test model

model.save('my_model.h5')
print('Model saved.')
loaded_model = tf.keras.models.load_model('my_model.h5')
print('Model loaded.')
test_loss, test_acc = loaded_model.evaluate(np.asarray(test_inputs), np.array(test_outputs))
print('Test accuracy:', test_acc)
