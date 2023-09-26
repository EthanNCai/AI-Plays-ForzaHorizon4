import cv2
import time
from grabscreen import grab_screen
import numpy as np

while True:
    # Grab the screen image
    screen = grab_screen(display_index=1, region=(200, 250, 700, 530))

    height, width, _ = screen.shape

    # Calculate midpoints coordinates
    left_mid = (0, height // 2)
    right_mid = (width - 1, height // 2)
    top_mid = (width // 2, 0)
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)

    # Create a mask image
    mask = np.zeros(screen.shape[:2], dtype=np.uint8)
    roi_corners = np.array([[left_mid, top_mid, right_mid, bottom_right, bottom_left]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    # Apply the mask


    # Apply erosion and dilation
    kernel = np.ones((3, 3), np.uint8)
    screen = cv2.erode(screen, kernel, iterations=1)
    screen = cv2.dilate(screen, kernel, iterations=1)

    # Edge detection
    edges = cv2.Canny(screen, 255, 255)
    roi = cv2.bitwise_and(edges, edges, mask=mask)
    # Display the edgesssss

    output = cv2.resize(roi, (200, 100))
    cv2.imshow('output', output)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break