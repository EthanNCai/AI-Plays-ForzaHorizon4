from Utilities.directkeys import PressKey, ReleaseKey, W, A, S, D
import time

t_time = 0.01


def key_press(key_output, confidence):
    if key_output[0] == 1:
        PressKey(W)
    if key_output[1] == 1:
        PressKey(A)
    if key_output[2] == 1:
        PressKey(S)
    if key_output[3] == 1:
        PressKey(D)
    time.sleep((1-confidence)*t_time)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
