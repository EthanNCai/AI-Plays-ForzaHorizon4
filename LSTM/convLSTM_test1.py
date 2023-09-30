import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense
import tensorflow as tf

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
past_frames = 3

# Step1 : load datasets
data = np.load('train_data.npy', allow_pickle=True)

inputs = np.array([item[0] for item in data])
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
    seq_x, seq_y = inputs[i:end_ix], outputs[i:end_ix]
    X.append(seq_x)
    y.append(seq_y)

# Step4 : Split data

train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(X, y, test_size=0.3,
                                                                          random_state=42)
# Step5 : Define model

model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=( 3, 3), activation='relu',
                     input_shape=(past_frames, rows, cols, 1),
                     padding='same',return_sequences=True, data_format='channels_last'))
model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False))
model.add(BatchNormalization())
model.add(Dense(16, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step6 : Train model

print(np.shape(train_inputs),np.shape(train_outputs))
model.fit(train_inputs, train_outputs, epochs=3,  validation_data=(test_inputs, test_outputs))

# Step7 : Test model

# model.save('my_model.h5')
# print('Model saved.')
# loaded_model = tf.keras.models.load_model('my_model.h5')
# print('Model loaded.')
# test_loss, test_acc = loaded_model.evaluate(test_inputs, test_outputs)
# print('Test accuracy:', test_acc)