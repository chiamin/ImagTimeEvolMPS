using ITensors, ITensorMPS

include("../Hamiltonian.jl")

function DMRG_TEST()
    Lx = 4
    Ly = 1
    tx = 1
    ty = 1
    U = 2
    tpr = 0
    tppx = 0
    tppy = 0
    mu = 0
    periodic_x = false
    periodic_y = false
    ampo = Hubbard(Lx, Ly, tx, ty, U, tpr, tppx, tppy, mu, periodic_x, periodic_y)


    N = Lx*Ly
    sites = siteinds("Electron",N,conserve_qns=true)

    H = MPO(ampo,sites)

    nsweeps = 5 # number of sweeps is 5
    maxdim = [10,20,40,100] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error

    states = RandomState(N; Nup=1, Ndn=1)
    psi0 = MPS(sites, states)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    return energy, psi
end

DMRG_TEST()

