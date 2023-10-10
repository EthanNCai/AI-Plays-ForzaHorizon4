def translate_wasd(lst):
    output = ''
    if lst[0] == 1:
        output += 'W'
    if lst[1] == 1:
        output += 'A'
    if lst[2] == 1:
        output += 'S'
    if lst[3] == 1:
        output += 'D'
    if len(output) == 0:
        output = 'nothing'
    return output
