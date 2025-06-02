import numpy as np
import matplotlib.pyplot as plt

N = 4 * 4  # 16

def readE0(fname):
    """
    Reads a file where each line starts with a name followed by one or more values.
    Values can be optionally enclosed in [ ] and may be separated by spaces or commas.
    Returns a dict where each name maps to either a float or a list of floats.
    """
    res = {}
    with open(fname, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            name = parts[0]
            value_str = parts[1].strip()

            # Remove optional [ and ]
            if value_str.startswith("[") and value_str.endswith("]"):
                value_str = value_str[1:-1].strip()

            # Split on commas or whitespace
            raw_values = [v for v in value_str.replace(",", " ").split() if v]

            values = [float(v) for v in raw_values]
            res[name] = values[0] if len(values) == 1 else values

    return res

def MCAnalysis(data, N_skip, plot=False):
    """Performs a Monte Carlo analysis on data:
       - Skips the first N_skip points,
       - Computes the running mean and standard error in steps of 10,
       - Optionally plots the errorbar."""
    data = data[N_skip:]
    assert len(data) > 0, "No data left after skipping."
    interval = 10
    vals = []
    errs = []
    # Note: starting at 2 up to len(data) inclusive (Python is 0-indexed)
    for i in range(2, len(data) + 1, interval):
        tmp = data[:i]
        val = np.mean(tmp)
        # Using ddof=1 to match Julia's sample standard deviation calculation
        err = np.std(tmp, ddof=1) / np.sqrt(i)
        vals.append(val)
        errs.append(err)
    if plot:
        plt.figure()
        xvals = np.arange(1, len(vals) + 1)
        plt.errorbar(xvals, vals, yerr=errs, fmt='o')
        plt.show()
    return np.array(vals), np.array(errs)

def geten(fname, N_skip):
    """Reads the data file (assumed to have four columns) and performs MCAnalysis on columns 2-4."""
    print(fname)
    data = np.loadtxt(fname)
    # Assuming the file has at least four columns: steps, Eks, EVs, signs
    # In Python, columns are 0-indexed.
    steps = data[:, 0]
    Eks = data[:, 1]
    EVs = data[:, 2]
    signs = data[:, 3]
    Eks_vals, errEks = MCAnalysis(Eks, N_skip)
    EVs_vals, errEVs = MCAnalysis(EVs, N_skip)
    signs_vals, err_signs = MCAnalysis(signs, N_skip)
    return Eks_vals, errEks, EVs_vals, errEVs, signs_vals, err_signs

def getDensity(fnames, N_skip):
    # Function left unimplemented, as in the original Julia code.
    pass

def divErr(A, errA, B, errB):
    """Propagate errors for a division: (A/B) with uncertainties errA and errB."""
    tmpA = (errA / A) ** 2
    tmpB = (errB / B) ** 2
    tmp = np.sqrt(tmpA + tmpB)
    return np.abs((A / B) * tmp)

def getAll(fnames, Ntaus, N_skip):
    """Loop over the provided file names, run geten on each,
       and combine the final values into arrays. Adjusts the energies by the signs and divides by N."""
    Ekst, errEkst = [], []
    EVst, errEVst = [], []
    signst, err_signst = [], []
    for fname in fnames:
        Eks_vals, errEks, EVs_vals, errEVs, signs_vals, err_signs = geten(fname, N_skip)
        Ekst.append(Eks_vals[-1])
        errEkst.append(errEks[-1])
        EVst.append(EVs_vals[-1])
        errEVst.append(errEVs[-1])
        signst.append(signs_vals[-1])
        err_signst.append(err_signs[-1])
    
    # Convert lists to numpy arrays for element-wise operations
    Ekst = np.array(Ekst)
    errEkst = np.array(errEkst)
    EVst = np.array(EVst)
    errEVst = np.array(errEVst)
    signst = np.array(signst)
    err_signst = np.array(err_signst)
    
    # Divide energies by the sign
    #print(Ekst)
    #print(signst)
    #Ekst = Ekst / signst
    #EVst = EVst / signst
    # Propagate errors through the division
    #errEkst = divErr(Ekst, errEkst, signst, err_signst)
    #errEVst = divErr(EVst, errEVst, signst, err_signst)
    
    # Divide by N
    Ekst = Ekst / N
    errEkst = errEkst / N
    EVst = EVst / N
    errEVst = errEVst / N
    
    return Ekst, errEkst, EVst, errEVst, signst, err_signst

if __name__ == "__main__":
    plt.ion()  # Turn on interactive mode
    data0 = readE0("init.dat")
    print(data0)
    
    fnames = ["en10.dat"]#, "en20.dat", "en30.dat"]
    Ntaus = [0, 10]
    taus = np.array(Ntaus) * 0.05  # taus in the original code
    
    N_skip = 0
    Ekst, errEkst, EVst, errEVst, signst, err_signst = getAll(fnames, Ntaus, N_skip)
    
    # Insert the initial values from data0 at the beginning
    Ekst = np.insert(Ekst, 0, data0["Ek0"])
    EVst = np.insert(EVst, 0, data0["EV0"])
    errEkst = np.insert(errEkst, 0, 0.0)
    errEVst = np.insert(errEVst, 0, 0.0)
    
    # Total energy and its error as the sum of kinetic and potential parts
    Est = Ekst + EVst
    errEst = errEkst + errEVst
    
    # Plot E_k
    f1 = plt.figure()
    print(len(taus), len(Ekst), len(errEkst))
    plt.errorbar(taus, Ekst, yerr=errEkst, marker="o", linestyle="-")
    plt.axhline(y=data0["Ek_GS"], linestyle="--", color="k")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"$E_k/N$", fontsize=18)
    plt.tight_layout()
    plt.savefig("Ek.pdf")
    
    # Plot E_V
    f2 = plt.figure()
    plt.errorbar(taus, EVst, yerr=errEVst, marker="o", linestyle="-")
    plt.axhline(y=data0["EV_GS"], linestyle="--", color="k")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"$E_V/N$", fontsize=18)
    plt.tight_layout()
    plt.savefig("EV.pdf")
    
    # Plot E (total energy)
    f3 = plt.figure()
    plt.errorbar(taus, Est, yerr=errEst, marker="o", linestyle="-")
    plt.axhline(y=data0["Ek_GS"] + data0["EV_GS"], linestyle="--", color="k")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"$E/N$", fontsize=18)
    plt.tight_layout()
    plt.savefig("E.pdf")
    
    # Plot sign values (excluding the initial tau=0 value)
    f4 = plt.figure()
    # In Julia, taus[2:end] corresponds to Python's taus[1:]
    plt.errorbar(taus[1:], signst, yerr=err_signst, marker="o", linestyle="-")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel("sign", fontsize=18)
    plt.tight_layout()
    plt.savefig("sign.pdf")
    
    plt.show(block=True)

