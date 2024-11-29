import Random
using LinearAlgebra
include("../H_k.jl")
include("../HSTrans.jl")
include("../DetTools.jl")
include("../SampleDet.jl")
include("../Hamiltonian.jl")
include("../dmrg.jl")
include("../SampleMPSDet.jl")

function getED0(L, pbc, t, U, dtau, Ntau, U0)
    psi, E0, Hk, HV, H = ED_GS(L, pbc, t, U0)
    _, E0, Hk, HV, H = ED_GS(L, pbc, t, U)
    expH = exp(-0.5*dtau*Hk)*exp(-dtau*HV)*exp(-0.5*dtau*Hk)
    psit = copy(psi)
    for i=1:Ntau
        psit = expH * psit
    end
    O = psit' * psi
    E = psit' * H * psi
    Ek = psit' * Hk * psi
    EV = psit' * HV * psi
    return E/O, Ek/O, EV/O, O
end

function getEkEV(mps, Lx, Ly, tx, ty, U, xpbc, ypbc)
    sites = siteinds(mps)
    ampo = Hubbard(Lx, Ly, tx, ty, 0., 0., 0., 0., 0., xpbc, ypbc)
    Hk = MPO(ampo,sites)
    ampo = Hubbard(Lx, Ly, 0., 0., U, 0., 0., 0., 0., xpbc, ypbc)
    HV = MPO(ampo,sites)
    Ek = inner(mps',Hk,mps)
    EV = inner(mps',HV,mps)
    return Ek, EV
end

function run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi)
    Nsites = Lx*Ly
    Npar = Nup+Ndn
    tau = dtau * nsteps

    # Initialize for QMC
    Hk, expHk_half, expV_up, expV_dn, auxflds = initQMC(Lx, Ly, 1, tx, ty, 0., U, xpbc, ypbc, false, dtau, nsteps, Nsites)
    Ntau = length(auxflds)

    # Initialize product states
    conf_beg = RandomConf(Nsites; Nup=Nup, Ndn=Ndn)
    conf_end = RandomConf(Nsites; Nup=Nup, Ndn=Ndn)
    phi1_up, phi1_dn = prodDetUpDn(conf_beg)
    phi2_up, phi2_dn = prodDetUpDn(conf_end)

    # Initialize all the determinants
    phis_up = initPhis(phi1_up, phi1_up, expHk_half, auxflds, expV_up)
    phis_dn = initPhis(phi2_up, phi2_dn, expHk_half, auxflds, expV_dn)

    # Initialize observables
    obs = makeObsDict()
    obs["Hk"] = Hk
    obs["U"] = U

    # Initialize MPS
    #en0, psi = Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps=100, maxdim=[10], cutoff=[1e-14])
    ppsi = makeProdMPS(psi)
    #println("E0 = ",en0)


    # Get exact energy from DMRG
    #=en_DMRG, psi_DMRG = Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps=100, maxdim=[20,20,20,20,40,40,40,40,80,80,80,80,160], cutoff=[1e-14])
    Ek0, EV0 = getEkEV(psi, Lx, Ly, tx, ty, U, xpbc, ypbc)
    Ek_DMRG, EV_DMRG = getEkEV(psi_DMRG, Lx, Ly, tx, ty, U, xpbc, ypbc)
    open("en0.dat","w") do file
        println(file,"E0 ",en0)
        println(file,"Ek0 ",Ek0)
        println(file,"EV0 ",EV0)
        println(file,"E_GS ",en_DMRG)
        println(file,"Ek_GS ",Ek_DMRG)
        println(file,"EV_GS ",EV_DMRG)
    end=#


    #E_GS, Ek_GS, EV_GS, O = getED0(Lx, xpbc, tx, U, dtau, Ntau)
    #println("ED E,Ek,EV ",E_GS," ", Ek_GS," ", EV_GS," ",O)

    det_time = 0.
    mps_time = 0.
    mea_time = 0.

    # Monte Carlo sampling
    latt = makeSquareLattice(Lx, Ly, xpbc, ypbc)
    for i=1:N_samples
        # Sample the left product state
        mps_time += @elapsed conf_beg, wMPS1 = sampleMPS!(conf_beg, ppsi, phis_up[1], phis_dn[1], latt)
        phis_up[0], phis_dn[0] = prodDetUpDn(conf_beg)

        # Sample the auxiliary fields from left to right
        det_time += @elapsed wDet = sampleAuxField_sweep!(phis_up, phis_dn, auxflds, expHk_half, expV_up, expV_dn; toRight=true)

        # Sample the right product state
        mps_time += @elapsed conf_end, wMPS2 = sampleMPS!(conf_end, ppsi, phis_up[Ntau], phis_dn[Ntau], latt)
        phis_up[Ntau+1], phis_dn[Ntau+1] = prodDetUpDn(conf_end)

        # Sample the auxiliary fields from right to left
        det_time += @elapsed wDet = sampleAuxField_sweep!(phis_up, phis_dn, auxflds, expHk_half, expV_up, expV_dn; toRight=false)

        # Measure
        w = wMPS1 * wDet * wMPS2
        mea_time += @elapsed measure!(phis_up[0], phis_dn[0], phis_up[1], phis_dn[1], w, obs)

        if i%100 == 0
            Eki = getObs(obs["Ek"])
            EVi = getObs(obs["EV"])
            si = getObs(obs["sign"])
            println(i," ",Eki/si," ",EVi/si," ",si)
            cleanObsDict(obs)
            println(det_time," ",mps_time," ",mea_time)
        end
    end
    println(det_time," ",mps_time)
end

function main()
    Lx=4
    Ly=4
    tx=ty=1.0
    xpbc=false
    ypbc=false
    Nup = 2
    Ndn = 2
    U = 12.
    dtau = 0.05
    nsteps = 10
    N_samples = 2000

    # Initialize MPS
    en0, psi = Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps=10, maxdim=[10], cutoff=[1e-14])
    println("E0 = ",en0)


    # Get exact energy from DMRG
    #=en_DMRG, psi_DMRG = Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps=100, maxdim=[20,20,20,20,40,40,40,40,80,80,80,80,160], cutoff=[1e-14])
    Ek0, EV0 = getEkEV(psi, Lx, Ly, tx, ty, U, xpbc, ypbc)
    Ek_DMRG, EV_DMRG = getEkEV(psi_DMRG, Lx, Ly, tx, ty, U, xpbc, ypbc)
    open("en0.dat","w") do file
        println(file,"E0 ",en0)
        println(file,"Ek0 ",Ek0)
        println(file,"EV0 ",EV0)
        println(file,"E_GS ",en_DMRG)
        println(file,"Ek_GS ",Ek_DMRG)
        println(file,"EV_GS ",EV_DMRG)
    end=#

    for nsteps in [40]
        run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi)
    end

end

main()
