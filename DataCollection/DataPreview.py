import numpy as np
import cv2


file_name = 'train_data.npy'
train_data = list(np.load(file_name, allow_pickle=True))

for data in train_data:
    img = data[0]
    target = data[1]
    cv2.imshow('test', img)
    print(target)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
