using ITensors, ITensorMPS

include("Hamiltonian.jl")

function Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps, maxdim, cutoff)
    N = Lx*Ly
    sites = siteinds("Electron", N, conserve_qns=true)
    ampo = Hubbard(Lx, Ly, tx, ty, U, 0, 0, 0, 0, xpbc, ypbc)
    H = MPO(ampo,sites)

    states = RandomState(N; Nup, Ndn)
    #states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = MPS(sites, states)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)
    return energy, psi
end
