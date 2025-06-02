import numpy as np
import analysis as ana
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/chiamin/mypy/")
import plotsetting as ps

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

if __name__ == "__main__":
    data0 = readE0("init.dat")
    skip_steps = 100

    fnames = ["ntau10.dat","ntau20.dat","ntau30.dat","ntau40.dat","ntau50.dat"]#,"ntau60.dat","ntau70.dat","ntau80.dat"]
    Ntaus = [0, 10, 20, 30, 40, 50]#, 60, 70, 80]
    taus = np.array(Ntaus) * 0.05  # taus in the original code

    Ekst, errEkst, EVst, errEVst, signst, err_signst = [data0["Ek0"]],[0.],[data0["EV0"]],[0.],[1.],[0.]
    for fil in fnames:
        data = ana.read_monte_carlo_file(fil, skip_steps=skip_steps)
        obs = ana.compute_mean_and_error_with_sign(data)
        Ekst.append(obs["Ek"][0])
        errEkst.append(obs["Ek"][1])
        EVst.append(obs["EV"][0])
        errEVst.append(obs["EV"][1])
        signst.append(obs["sign"][0])
        err_signst.append(obs["sign"][1])
    Ekst, errEkst, EVst, errEVst, signst, err_signst = map(np.array, [Ekst, errEkst, EVst, errEVst, signst, err_signst])

    # Divide by N
    N = 2*2
    Ekst = Ekst / N
    errEkst = errEkst / N
    EVst = EVst / N
    errEVst = errEVst / N

    # Total energy and its error as the sum of kinetic and potential parts
    Est = Ekst + EVst
    errEst = errEkst + errEVst
    
    # Plot E_k
    f1 = plt.figure()
    plt.errorbar(taus, Ekst, yerr=errEkst, marker="o", linestyle="-")
    plt.axhline(y=data0["Ek_GS"]/N, linestyle="--", color="k")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"$E_k/N$", fontsize=18)
    plt.tight_layout()
    
    # Plot E_V
    f2 = plt.figure()
    plt.errorbar(taus, EVst, yerr=errEVst, marker="o", linestyle="-")
    plt.axhline(y=data0["EV_GS"]/N, linestyle="--", color="k")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"$E_V/N$", fontsize=18)
    plt.tight_layout()
    
    # Plot E (total energy)
    f3 = plt.figure()
    plt.errorbar(taus, Est, yerr=errEst, marker="o", linestyle="-")
    plt.axhline(y=(data0["Ek_GS"] + data0["EV_GS"])/N, linestyle="--", color="k")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"$E/N$", fontsize=18)
    plt.tight_layout()
    
    # Plot sign values (excluding the initial tau=0 value)
    f4 = plt.figure()
    # In Julia, taus[2:end] corresponds to Python's taus[1:]
    plt.errorbar(taus, signst, yerr=err_signst, marker="o", linestyle="-")
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel("sign", fontsize=24)
    plt.tight_layout()

    # Compare with ED
    data = np.loadtxt('enED.txt', skiprows=1)
    taus = data[:, 0]
    Es = data[:, 1]
    Eks = data[:, 2]
    EVs = data[:, 3]
    ax1 = f1.axes[0]
    ax1.plot(taus,Eks,c='r',label="ED")
    ax2 = f2.axes[0]
    ax2.plot(taus,EVs,c='r',label="ED")
    ax3 = f3.axes[0]
    ax3.plot(taus,Es,c='r',label="ED")

    # Compare with ED using Trotter decompositions
    data = np.loadtxt('enTrotter.txt', skiprows=1)
    taus = data[:, 0]
    Es = data[:, 1]
    Eks = data[:, 2]
    EVs = data[:, 3]
    ax1 = f1.axes[0]
    ax1.plot(taus,Eks,c='orange',label="Trotter")
    ax2 = f2.axes[0]
    ax2.plot(taus,EVs,c='orange',label="Trotter")
    ax3 = f3.axes[0]
    ax3.plot(taus,Es,c='orange',label="Trotter")
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ps.set((ax1,ax2,ax3))

    f1.savefig("Ek.pdf")
    f2.savefig("EV.pdf")
    f3.savefig("E.pdf")
    f4.savefig("sign.pdf")
    plt.show(block=True)
