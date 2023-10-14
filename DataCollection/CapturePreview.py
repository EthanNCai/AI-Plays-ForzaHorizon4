import cv2
from Utilities.grabscreen import grab_screen
from Utilities.cv_crop_processing import crop_screen
from Utilities.cv_edge_processing import edge_processing
import numpy as np

while True:
    # grab the screen image
    screen = grab_screen(display_index=1, region=(0, 0, 1280, 720))
    screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # cropping the image
    cropped_screen = crop_screen(screen, trim_rate=0.3)

    resized_image = edge_processing(cropped_screen, resize_width=200, resize_height=100)

    cv2.imshow('original', screen_rgb)
    cv2.imshow('processed', resized_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
