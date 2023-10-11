import cv2
from Utilities.grabscreen import grab_screen
from Utilities.cv_crop_processing import crop_screen
import numpy as np

trim_rate = 0.2

while True:
    # grab the screen image
    screen = grab_screen(display_index=1, region=(0, 0, 1280, 720))
    screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # cropping the image
    cropped_screen = crop_screen(screen, trim_rate)

    # convert to gray scale & resize
    cropped_gray = cv2.cvtColor(cropped_screen, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(cropped_gray, (300, 150))

    cv2.imshow('original', screen_rgb)
    cv2.imshow('processed', resized_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
