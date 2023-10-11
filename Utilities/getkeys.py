# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import numpy as np

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def key_to_wasd_format(keys_in):
    # [W,A,S,D]
    key_to_output = np.array([0, 0, 0, 0], dtype=np.uint8)
    if 'W' in keys_in:
        key_to_output[0] = 1
    if 'A' in keys_in:
        key_to_output[1] = 1
    if 'S' in keys_in:
        key_to_output[2] = 1
    if 'D' in keys_in:
        key_to_output[3] = 1
    return key_to_output
