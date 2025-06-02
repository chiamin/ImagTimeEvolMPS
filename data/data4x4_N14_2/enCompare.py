import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/chiamin/mypy/")
import plotsetting as ps

data1 = np.loadtxt('../data4x4_N14/en.txt', skiprows=1)
data2 = np.loadtxt('en.txt', skiprows=1)

dataGS = np.loadtxt('enGS.txt', skiprows=1)
Ek_GS, EV_GS, E_GS = dataGS

def plot(data, ax1, ax2, ax3, ax4, legend=""):
    taus = data[:,0]
    Ekst = data[:,1]
    errEkst = data[:,2]
    EVst = data[:,3]
    errEVst = data[:,4]
    Est = data[:,5]
    errEst = data[:,6]
    signst = data[:,7]
    err_signst = data[:,8]

    # Plot E_k
    ax1.errorbar(taus, Ekst, yerr=errEkst, marker="o", linestyle="-", label=legend)
    ax1.set_xlabel(r"$\tau$", fontsize=18)
    ax1.set_ylabel(r"$E_k/N$", fontsize=18)

    # Plot E_V
    ax2.errorbar(taus, EVst, yerr=errEVst, marker="o", linestyle="-", label=legend)
    ax2.set_xlabel(r"$\tau$", fontsize=18)
    ax2.set_ylabel(r"$E_V/N$", fontsize=18)

    # Plot E (total energy)
    ax3.errorbar(taus, Est, yerr=errEst, marker="o", linestyle="-", label=legend)
    ax3.set_xlabel(r"$\tau$", fontsize=18)
    ax3.set_ylabel(r"$E/N$", fontsize=18)

    # Plot sign values (excluding the initial tau=0 value)
    ax4.errorbar(taus, signst, yerr=err_signst, marker="o", linestyle="-", label=legend)
    ax4.set_xlabel(r"$\tau$", fontsize=18)
    ax4.set_ylabel("sign", fontsize=24)


f1,ax1 = plt.subplots()
f2,ax2 = plt.subplots()
f3,ax3 = plt.subplots()
f4,ax4 = plt.subplots()

plot(data1, ax1, ax2, ax3, ax4, legend="$D=4$")
plot(data2, ax1, ax2, ax3, ax4, legend="$D=80$")

ax1.axhline(Ek_GS, linestyle="--", color="k")
ax2.axhline(EV_GS, linestyle="--", color="k")
ax3.axhline(E_GS, linestyle="--", color="k")

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ps.set((ax1,ax2,ax3,ax4))

f1.savefig("Ek_compare.pdf")
f2.savefig("EV_compare.pdf")
f3.savefig("E_compare.pdf")
f4.savefig("sign_compare.pdf")
plt.show(block=True)
