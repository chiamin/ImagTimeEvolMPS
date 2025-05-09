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
    configs = []

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
            configs.append(arr)  # store original configuration

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    y_normalized = y_vals / y_vals.sum()

    return x_vals, y_normalized, configs

inds1, w1, conf1 = read_and_convert("ntau50_conf1.txt")
inds2, w2, conf2 = read_and_convert("ntau50_conf2.txt")

sorted_conf1 = [a for b, a in sorted(zip(w1, conf1), reverse=True)]
sorted_conf2 = [a for b, a in sorted(zip(w2, conf2), reverse=True)]
for i in sorted_conf1[:5]:
    print(np.reshape(i,(2,2)),"\n")
print("-------------------------------------------")
for i in sorted_conf2[:5]:
    print(np.reshape(i,(2,2)),"\n")
# Plot
plt.scatter(inds1, w1, marker='x')
plt.scatter(inds2, w2, marker='.')
plt.show()

