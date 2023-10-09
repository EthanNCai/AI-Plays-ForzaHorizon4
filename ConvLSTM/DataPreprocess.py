import numpy as np
from collections import Counter
import pandas as pd
import statistics
import Utilities.onehot
import random

rows = 100
cols = 200
channels = 1
past_frames = 7

# Step1 : load datasets
data = np.load('Files/train_data.npy', allow_pickle=True)
inputs = np.asarray([item[0] for item in data])
outputs = np.array([item[1] for item in data])

# Step2 : one-hot encode the label
outputs = np.array(Utilities.onehot.onehot_encode(outputs))
# inputs = np.expand_dims(inputs, axis=-1)

# Step3 : turn into sequential data

X, y = [], []
for i in range(len(inputs)):
    end_ix = i + past_frames
    if end_ix > len(inputs) - 1:
        break
    seq_x, seq_y = inputs[i:end_ix], outputs[end_ix]
    X.append(seq_x)
    y.append(seq_y)


# Step4 : balancing data

unbalanced_data = [[a, b] for a, b in zip(X, y)]
counter = Counter(map(tuple, pd.DataFrame(unbalanced_data)[1]))
counter_list = [(label, count) for label, count in counter.items()]
balancing_target = int(statistics.mean(count for _, count in counter_list))

balanced_data = []
for label, count in counter_list:
    label_data = [[i, l] for i, l in unbalanced_data if (label == l).all()]
    if count < balancing_target:
        balanced_data += label_data
    else:
        balanced_data += random.sample(label_data, balancing_target)

random.shuffle(balanced_data)

# Step5 : save to file

np.save('Files/preprocessed_data.npy', np.array(balanced_data, dtype=object))

# why reject

