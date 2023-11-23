import os.path
import cv2
import time
import numpy as np
from Utilities.grabscreen import grab_screen
from Utilities.getkeys import key_check, key_to_wasd_format
from Utilities.cv_crop_processing import crop_screen
from Utilities.cv_edge_processing import edge_processing
import time

# cv2 text configurations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_color = (255, 255, 255)
thickness = 1

# model configurations
serial_number = 2
file_name = "Files/dataset-" + str(serial_number) + ".npy"


# for monitoring fps
time_checkpoint_a, time_checkpoint_b = 0, 0

# Step 1: load or create a dataset file
if os.path.isfile(file_name):
    print('Dataset exists, appending...')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('Dataset does not exist, starting anew...')
    training_data = []

# Counting down 5 secs.
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)

# Target FPS
target_fps = 15
delay = 1 / target_fps

while True:
    time_checkpoint_a = time.time()

    # Step 2: Acquire the current frame and crop the edge.
    # Notice: Set this index to the physical display that your game window is on.
    screen = grab_screen(display_index=1, region=(0, 0, 1280, 720))
    screen_output = crop_screen(screen, trim_rate=0.3)

    # Step 3: Acquire the current key-press and interpret it into the [W, A, S, D] format
    keys = key_check()
    key_output = key_to_wasd_format(keys)

    # Step 4: Append this new record to the dataset
    training_data.append([screen_output, key_output])

    # Save the dataset every 2000 samples
    if len(training_data) % 2000 == 0:
        print(len(training_data))
        np.save(file_name, np.array(training_data, dtype=object))

    # Step 5: display the frame review

    # calculate the raw FPS
    time_checkpoint_b = time.time()
    raw_elapsed_time = time_checkpoint_b - time_checkpoint_a
    # delay if FPS too large, in order for FPS stay close to the target
    if raw_elapsed_time < delay:
        time.sleep(delay - raw_elapsed_time)
    # calculate the actual FPS
    time_checkpoint_c = time.time()
    actual_elapsed_time = time_checkpoint_c - time_checkpoint_a
    fps = 1 / actual_elapsed_time

    # put information on the screen (FPS, Total frames)
    screen_output_rgb = edge_processing(screen_output, resize_width=300, resize_height=150)
    cv2.putText(screen_output_rgb, 'FPS: {:.2f}'.format(fps),
                (10, 10), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(screen_output_rgb, 'Total Frames: {}'.format(len(training_data)),
                (10, 20), font, font_scale, font_color, thickness, cv2.LINE_AA)

    print('\rfps: {:.2f}'.format(fps),
          'total Frames: {}'.format(len(training_data)),
          end='')

    cv2.imshow('preview', screen_output_rgb)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
