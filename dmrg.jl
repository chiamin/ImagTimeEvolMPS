using ITensors, ITensorMPS

include("Hamiltonian.jl")

function Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn, psi0::MPS=MPS(); nsweeps, maxdim, cutoff)
    N = Lx*Ly
    if length(psi0) == 0
        sites = siteinds("Electron", N, conserve_qns=true)
    else
        sites = siteinds(psi0)
    end
    ampo = Hubbard(Lx, Ly, tx, ty, U, 0, 0, 0, 0, xpbc, ypbc)
    H = MPO(ampo,sites)

    if length(psi0) == 0
        states = RandomState(N; Nup, Ndn)
        #states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
        psi0 = MPS(sites, states)
    end
    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)
    return energy, psi
end
