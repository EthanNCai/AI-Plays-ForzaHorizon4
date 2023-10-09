import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import Utilities.onehot

# load data
data = np.load('Files/balanced_data.npy', allow_pickle=True)
inputs = np.array([item[0] for item in data])
outputs = np.array([item[1] for item in data])

# one-hot encode the output and adjust input shape to (w,h,depth)
outputs = np.array(Utilities.onehot.onehot_encode(outputs))
inputs = np.expand_dims(inputs, axis=-1)

# train data set and test data set splitting
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.3,
                                                                          random_state=42)
# model construction
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3])),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
print(np.shape(train_inputs), np.shape(train_outputs))
model.fit(train_inputs, train_outputs, epochs=3, batch_size=128, validation_data=(test_inputs, test_outputs))

# load and test
model.save('my_model.h5')
loaded_model = tf.keras.models.load_model('my_model.h5')
test_loss, test_acc = loaded_model.evaluate(test_inputs, test_outputs)
print('Test accuracy:', test_acc)
