import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2


train_data = list(np.load('train_data.npy', allow_pickle=True))
df = pd.DataFrame(train_data)
counter = Counter(map(tuple, df[1]))
balanced_count = min(counter[(1, 0, 0, 1)],counter[(1, 1, 0, 0)],counter[(1, 0, 0, 0)])*5

forward = []
left = []
right = []
other = []

for data in train_data:
    img = data[0]
    choice = data[1]
    if (choice == [1, 0, 0, 0]).all():
        forward.append([img, choice])
    elif (choice == [1, 1, 0, 0]).all():
        left.append([img, choice])
    elif (choice == [1, 0, 0, 1]).all():
        right.append([img, choice])
    else:
        other.append([img, choice])


if balanced_count < len(forward):
    forward = forward[:balanced_count]
if balanced_count < len(left):
    forward = forward[:balanced_count]
if balanced_count < len(right):
    forward = forward[:balanced_count]
if balanced_count < len(other):
    forward = forward[:balanced_count]

final_list = forward + left + right + other

shuffle(final_list)
f_df = pd.DataFrame(final_list)
print(Counter(f_df[1].apply(str)))
np.save('balanced_train_data.npy', np.array(final_list, dtype=object))









