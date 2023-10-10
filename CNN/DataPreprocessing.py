import numpy as np
import pandas as pd
from collections import Counter
import statistics
import random
import Utilities.onehot

# load data
data = list(np.load('Files/train_data.npy', allow_pickle=True))

inputs = np.asarray([item[0] for item in data])
outputs = np.array([item[1] for item in data])

inputs = np.expand_dims(inputs, axis=-1)
outputs = np.array(Utilities.onehot.onehot_encode(outputs))

unbalanced_data = [[a, b] for a, b in zip(inputs, outputs)]
# counting label & decide balancing target
counter = Counter(map(tuple, pd.DataFrame(unbalanced_data)[1]))
counter_list = [(label, count) for label, count in counter.items()]
balancing_target = int(statistics.mean(count for _, count in counter_list))

print('original_data:')
print(Counter(map(tuple, pd.DataFrame(data)[1])))

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

print('encoded_balanced_data')
print(Counter(map(tuple, pd.DataFrame(balanced_data)[1])))

np.save('Files/preprocessed_data.npy', np.array(balanced_data, dtype=object))
