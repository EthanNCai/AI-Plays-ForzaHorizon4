import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


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


# 加载数据
data = np.load('balanced_train_data.npy', allow_pickle=True)

# 分割输入和输出
inputs = np.array([item[0] for item in data])
outputs = np.array([item[1] for item in data])

outputs = np.array(encode_list(outputs))
real_outputs = []
for item in outputs:
    encoding = one_hot_encode(item, 16)
    real_outputs.append(encoding)
outputs = np.array(real_outputs)

# 划分训练集和测试集
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.3,
                                                                          random_state=42)
# 构建卷积神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 200, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 将输入数据调整为模型需要的形状
train_inputs = np.expand_dims(train_inputs, axis=-1)
test_inputs = np.expand_dims(test_inputs, axis=-1)
# 训练模型
model.fit(train_inputs, train_outputs, epochs=3, batch_size=128, validation_data=(test_inputs, test_outputs))

# 保存模型
model.save('my_model.h5')
print('Model saved.')

# 加载模型
loaded_model = tf.keras.models.load_model('my_model.h5')
print('Model loaded.')

test_loss, test_acc = loaded_model.evaluate(test_inputs, test_outputs)
print('Test accuracy:', test_acc)

plt.imshow(test_inputs[0])
