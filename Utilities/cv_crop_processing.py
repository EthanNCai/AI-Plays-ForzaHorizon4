# this function is for cropping the edge part of a screen(in order to get rid of the game UI).
# screen_in is the opencv screen array
# trim_rate is how much you want to cut, it a percentage number.

def crop_screen(screen_in, trim_rate=0.1):
    height, width, _ = screen_in.shape
    padding_w = int(width * trim_rate)
    padding_h = int(height * trim_rate)
    return screen_in[padding_h:-padding_h, padding_w:-padding_w, :]
