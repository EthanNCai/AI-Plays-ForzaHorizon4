import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import Utilities.onehot

model_path = 'Files/cv_cnn_model.h5'

# Step1 : load data
data = np.load('Files/preprocessed_data.npy', allow_pickle=True)
inputs = np.array([item[0] for item in data])
outputs = np.array([item[1] for item in data])

# Step2 : dataset splitting
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.3,
                                                                          random_state=42)
# Step3 : model construction
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

# Step4 : model training

print('input  shape:', train_inputs.shape)
print('output shape:', train_outputs.shape)

model.fit(train_inputs, train_outputs, epochs=3, batch_size=128, validation_data=(test_inputs, test_outputs))

# Step5 : save & load & test

model.save(model_path)
print('Model Saved')
loaded_model = tf.keras.models.load_model(model_path)
print('Model Loaded')
test_loss, test_acc = loaded_model.evaluate(test_inputs, test_outputs)
print('Val. accuracy:', test_acc)
