import matplotlib.pyplot as plt
import numpy as np

def array_to_number2(arr):
    result = 0
    base = 3  # Base-3 to differentiate 1 and 2 properly
    for i, val in enumerate(reversed(arr)):
        result += val * (base ** i)
    return result

def array_to_number4(arr):
    base4_digits = [x - 1 for x in arr]  # Convert [1,4] to [0,3]
    num = 0
    for d in base4_digits:
        num = num * 4 + d
    return num

def read_and_convert(filename):

    x_vals = []
    y_vals = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(']')
            arr_str = parts[0].strip('[').strip()

            # Replace commas with spaces, then split
            arr_elements = arr_str.replace(',', ' ').split()
            arr = list(map(int, arr_elements))

            value = float(parts[1].strip())

            num = array_to_number4(arr)

            x_vals.append(num)
            y_vals.append(value)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    y_normalized = y_vals / y_vals.sum()

    return x_vals, y_normalized

conf1, w1 = read_and_convert("/home/chiamin/project/2024/imagMPS/code/ED/w.txt")
conf2, w2 = read_and_convert("hist.txt")
# Plot
plt.scatter(conf1, w1, marker='x')
plt.scatter(conf2, w2, marker='.')
plt.show()

