import numpy as np
import ast

def parse_line_with_vectors(header, line):
    """
    Parses a single line of data, handling both scalar and vector observables.
    Vector values are assumed to be in the format: [x1, x2, ..., xn].
    """
    values = line.strip().split()
    result = []
    i = 0
    while i < len(values):
        if values[i].startswith('['):
            # Accumulate until the closing bracket is found
            vec = values[i]
            while not values[i].endswith(']'):
                i += 1
                vec += ' ' + values[i]
            result.append(ast.literal_eval(vec))  # Parse string to list
        else:
            result.append(float(values[i]))  # Scalar value
        i += 1
    return dict(zip(header, result))

def read_monte_carlo_file(file_path, skip_steps=0):
    """
    Reads a Monte Carlo data file with a header row and returns a dictionary of observables.
    Each observable is mapped to a NumPy array of values (scalars or vectors).
    Allows skipping a number of initial steps for thermalization.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split()  # First row: observable names
    data = {key: [] for key in header}

    # Start reading after skipping 'skip_steps' lines
    for line in lines[1 + skip_steps:]:
        parsed_line = parse_line_with_vectors(header, line)
        for key in header:
            data[key].append(parsed_line[key])

    # Convert lists to NumPy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data

def compute_mean_and_error_with_sign(data_dict):
    """
    Computes the sign-corrected mean and statistical error for each observable,
    including the 'sign' observable itself.
    Proper error propagation is applied.
    Returns a dictionary: observable -> (mean, error)
    """

    result = {}

    # Extract sign data
    sign = np.array(data_dict["sign"])

    # Compute the average sign ⟨sign⟩
    mean_sign = np.mean(sign)

    # Standard error of ⟨sign⟩ (sigma / sqrt(N))
    sign_error = np.std(sign, ddof=1) / np.sqrt(len(sign))

    # Always include ⟨sign⟩ itself in the results
    result["sign"] = (mean_sign, sign_error)

    # Loop through all other observables
    for key, values in data_dict.items():
        if key in ["step", "sign"]:
            continue  # Skip non-physical data

        values = np.array(values)  # Convert to NumPy array if not already

        # Compute average of x/sign
        mean_corr = np.mean(values, axis=0)

        # Standard error of x/sign
        std_corr = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))

        # Final observable: ⟨x/sign⟩ / ⟨sign⟩
        final_mean = mean_corr / mean_sign

        # Error propagation for:
        # f = a / b ⇒ σ_f^2 = (σ_a / b)^2 + (a * σ_b / b^2)^2
        final_error = np.sqrt(
            (std_corr / mean_sign) ** 2 +
            (mean_corr * sign_error / mean_sign**2) ** 2
        )

        # Format output nicely
        if np.isscalar(final_mean) or np.ndim(final_mean) == 0:
            # Convert scalar numpy floats to native Python float
            result[key] = (float(final_mean), float(final_error))
        else:
            # For vectors, return NumPy arrays
            result[key] = (np.array(final_mean), np.array(final_error))

    return result


# Example usage with the uploaded file
if __name__ == "__main__":
    file_path = "ntau10.dat"  # Ensure this file is in the same directory
    data = read_monte_carlo_file(file_path)
    results = compute_mean_and_error_with_sign(data)

    # Print the results for each observable
    for key, (mean, error) in results.items():
        print(f"{key}:")
        print(f"  Mean  = {mean}")
        print(f"  Error = {error}")
        print()

