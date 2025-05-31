import Random
using LinearAlgebra
include("../H_k.jl")
include("../HSTrans.jl")
include("../DetTools.jl")
include("../SampleDet.jl")
include("../EDTwoParticles.jl")

function SampleDet_Test()
    Lx=4
    Ly=1
    tx=ty=1.0
    xpbc=false
    ypbc=false
    Nup = 1
    Ndn = 1
    U = 8.0
    Nsites = Lx*Ly
    Npar = Nup+Ndn
    tau = 0.2
    nsteps = 4
    dtau = tau/nsteps
    N_samples = 400000

    Hk, expHk_half, expV_up, expV_dn, auxflds = initQMC(Lx, Ly, 1, tx, ty, 0., U, xpbc, ypbc, false, dtau, nsteps, Nsites)


    F = eigen(Hk)
    phi_up = F.vectors[:,1:Nup]
    phi_dn = F.vectors[:,1:Ndn]

    phis_up = initPhis(phi_up, phi_up, expHk_half, auxflds, expV_up)
    phis_dn = initPhis(phi_dn, phi_dn, expHk_half, auxflds, expV_dn)


    #------------------------- ED -----------------------------
    psi = two_par_state(phi_up, phi_dn)
    HkED = -tx * H_hop(Nsites, xpbc)
    HVED = U * H_nn(Nsites)
    HED = HkED + HVED
    expHV = exp(-dtau*HVED)
    expHk = exp(-0.5*dtau*HkED)
    expH = expHk * expHV * expHk
    psiED = psi
    for i=1:nsteps
        psiED = expH * psiED
    end
    EV = psiED' * HVED * psiED
    Ek = psiED' * HkED * psiED
    O = psiED' * psiED
    #----------------------------------------------------------


    # Monte Carlo sampling
    obs = makeObsDict()
    obs["Hk"] = Hk
    obs["U"] = U
    for i=1:N_samples
        sampleAuxField_sweep!(phis_up, phis_dn, auxflds, expHk_half, expV_up, expV_dn, obs; toRight=true)
        sampleAuxField_sweep!(phis_up, phis_dn, auxflds, expHk_half, expV_up, expV_dn, obs; toRight=false)

        if i%10000 == 0
            Eki = getObs(obs["Ek"])
            EVi = getObs(obs["EV"])
            si = getObs(obs["sign"])
            println(i,"  Ek=",Eki/si,"  EV=",EVi/si,"  sign=",si)
        end
    end

    println("ED Ek = ",Ek/O)
    println("ED EV = ",EV/O)
end

SampleDet_Test()
