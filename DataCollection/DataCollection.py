import os.path
import cv2
import time
import numpy as np
from Utilities.grabscreen import grab_screen
from Utilities.getkeys import key_check, key_to_wasd_format
from Utilities.cv_crop_processing import crop_screen


serial_number = 0
file_name = "dataset-" + str(serial_number) + "-.npy"
trim_rate = 0.2

# Step1 : load or create a dataset file
if os.path.isfile(file_name):
    print('dataset exist, appending .....')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('dataset not exist, starting anew .....')
    training_data = []

# counting down 10 secs.
for i in list(range(10))[::-1]:
    print(i + 1)
    time.sleep(1)

while True:

    # Step2 : acquire the current frame and crop the edge.
    # Notice : set this index to the physical display that your game window at
    screen = grab_screen(display_index=1, region=(0, 0, 1280, 720))
    screen_output = crop_screen(screen, trim_rate)

    # Step3 : acquire the current key-press and interpret it into the [W,A,S,D] format
    keys = key_check()
    key_output = key_to_wasd_format(keys)

    # Step4 : append this new record into the dataset
    training_data.append([screen_output, key_output])

    screen_output_rgb = cv2.cvtColor(screen_output, cv2.COLOR_BGR2RGB)
    cv2.imshow('preview', screen_output_rgb)

    # save the dataset every 500 samples.
    if len(training_data) % 500 == 0:
        print(len(training_data))
        np.save(file_name, np.array(training_data, dtype=object))

    # some cv ritual
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
