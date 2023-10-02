import numpy as np
import pandas as pd
from collections import Counter
import statistics
import random

# load data
unbalanced_data = list(np.load('./train_data.npy', allow_pickle=True))

# counting label & decide balancing target
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
print('balanced_data')
print(Counter(pd.DataFrame(balanced_data)[1].apply(str)))

np.save('balanced_data', np.array(balanced_data, dtype=object))







