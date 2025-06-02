import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def read_and_convert(filename):
    def array_to_number(arr):
        base4_digits = [x - 1 for x in arr]  # Convert [1,4] to [0,3]
        num = 0
        for d in base4_digits:
            num = num * 4 + d
        return num

    x_vals = []
    y_vals = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(']')
            arr_str = parts[0].strip('[').strip()
            arr = list(map(int, arr_str.split(',')))
            value = float(parts[1].strip())
            
            num = array_to_number(arr)
            
            x_vals.append(num)
            y_vals.append(value)
    
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    y_normalized = y_vals / y_vals.sum()

    return x_vals, y_normalized

conf1, w1 = read_and_convert("w.txt")
conf2, w2 = read_and_convert("hist.txt")
# Plot
plt.scatter(conf1, w1, marker='x')
plt.scatter(conf2, w2, marker='.')
plt.show()

