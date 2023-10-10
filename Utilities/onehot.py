import numpy as np
import math


def generate_binary_numbers(num_bits):
    if num_bits <= 0:
        return []
    binary_numbers = []
    max_number = 2 ** num_bits
    for i in range(max_number):
        binary = bin(i)[2:].zfill(num_bits)
        binary_list = [int(bit) for bit in binary]
        binary_numbers.append(binary_list)
    return binary_numbers


def generate_onehot_list(num_bits):
    onehot_list = []
    for i in range(num_bits ** 2):
        temp = [int(i == j) for j in range(num_bits ** 2)]
        onehot_list.append(temp)
    return onehot_list


def onehot_encode(lst):
    digits_in = len(lst[0])
    bin_list = generate_binary_numbers(digits_in)
    one_hot_list = generate_onehot_list(digits_in)
    dictionary = {tuple(bin_list[i]): one_hot_list[i] for i in range(len(bin_list))}
    return [dictionary[tuple(item)] for item in lst]


def onehot_decode(lst):
    digits_in = int(math.log(len(lst[0]), 2))
    bin_list = generate_binary_numbers(digits_in)
    one_hot_list = generate_onehot_list(digits_in)
    dictionary = {tuple(one_hot_list[i]): bin_list[i] for i in range(len(bin_list))}
    return [dictionary[tuple(item)] for item in lst]
