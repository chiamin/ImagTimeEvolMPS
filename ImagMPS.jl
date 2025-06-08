import Random
using LinearAlgebra
include("HSTrans.jl")
include("DetTools.jl")
include("SampleDet.jl")
include("Hamiltonian.jl")
include("SampleMPSDet.jl")
include("Initial.jl")
include("Timer.jl")
include("help.jl")
using ITensorMPS

seed = time_ns()
Random.seed!(seed)
println("Random number seed: ",seed)

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

function run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, mps, write_step, dir)
    Nsites = Lx*Ly
    Npar = Nup+Ndn
    tau = dtau * nsteps

    # Initialize for QMC
    Hk, expV_up, expV_dn, auxflds = initQMC(Lx, Ly, tx, ty, U, xpbc, ypbc, dtau, nsteps, Nsites)
    expHk = exp(-dtau*Hk)
    expHk_half = exp(-0.5*dtau*Hk)
    expHk_half_inv = exp(+0.5*dtau*Hk)
    Ntau = length(auxflds)

    # Initialize product states by sampling the MPS
    conf_beg = ITensorMPS.sample(mps)
    conf_end = deepcopy(conf_beg)
    println("Initial conf: ",conf_beg," ",conf_end)
    phi1_up, phi1_dn = prodDetUpDn(conf_beg)
    phi2_up, phi2_dn = prodDetUpDn(conf_end)

    open(dir*"/init.dat","a") do file
        println(file,"Initial_conf: ",conf_beg," ",conf_end)
    end
        
    # Compute the overlaps
    OMPS1 = MPSOverlap(conf_beg, mps)
    OMPS2 = MPSOverlap(conf_end, mps)

    # Initialize all the determinants
    phiL_up = initPhis(phi1_up, expHk, expHk_half, auxflds, expV_up)
    phiR_up = initPhis(phi2_up, expHk, expHk_half, reverse(auxflds), expV_up)
    phiL_dn = initPhis(phi1_dn, expHk, expHk_half, auxflds, expV_dn)
    phiR_dn = initPhis(phi2_dn, expHk, expHk_half, reverse(auxflds), expV_dn)
    @assert length(phiL_up) == Ntau + 1
    @assert length(phiR_up) == Ntau + 1
    @assert length(phiL_dn) == Ntau + 1
    @assert length(phiR_dn) == Ntau + 1

    # Initialize observables
    obs = Dict{String,Any}()

    # Store some objects that will be used in measurement
    para = Dict{String,Any}()
    para["Hk"] = Hk
    para["U"] = U

    # Initialize MPS machine which is efficient in computing the overlap with a product state
    mpsM = makeProdMPS(mps)

    # Reset the timer
    treset()

    file = open(dir*"/ntau"*string(nsteps)*".dat","w")
    # Write the observables' names
    println(file,"step Ek EV E sign nup ndn")



    # Monte Carlo sampling
    latt = makeSquareLattice(Lx, Ly, xpbc, ypbc)
    c = div(Ntau,2)
    for iMC=1:N_samples
        # 1. Sample the left product state
        #    OMPS1: <MPS|conf1>
        tstart("MPS")
        conf_beg, OMPS1 = sampleMPS!(conf_beg, mpsM, phiR_up[end], phiR_dn[end], latt)
        # Update phiL[1]
        phi_up, phi_dn = prodDetUpDn(conf_beg)
        phiL_up[1] = expHk_half * phi_up
        phiL_dn[1] = expHk_half * phi_dn
        tend("MPS")
        #@assert abs(OMPS1-MPSOverlap(conf_beg, mps)) < 1e-14    # Check MPS overlap

        # 2. Sample the auxiliary fields from left to right
        #    ODet: <conf1|BB...B|conf2>
        tstart("Det")
        for i=1:Ntau
            # Sample the fields
            phi_up, phi_dn, ODet, auxflds[i] = sampleAuxField(phiL_up[i], phiL_dn[i], phiR_up[end-i], phiR_dn[end-i],
                                                              expV_up, expV_dn, auxflds[i], true)
            # Propagate B_K
            if i == Ntau
                phiL_up[i+1] = expHk_half * phi_up
                phiL_dn[i+1] = expHk_half * phi_dn
            else
                phiL_up[i+1] = expHk * phi_up
                phiL_dn[i+1] = expHk * phi_dn
            end

            # Measure at the center slice
            if (i == c)
                O = ODet * conj(OMPS1) * OMPS2
                phiLc_up = expHk_half * phi_up
                phiLc_dn = expHk_half * phi_dn
                phiRc_up = expHk_half_inv * phiR_up[end-i]
                phiRc_dn = expHk_half_inv * phiR_dn[end-i]
                measure!(phiLc_up, phiLc_dn, phiRc_up, phiRc_dn, sign(O), obs, para)
            end
        end
        tend("Det")


        # 3. Sample the right product state
        #    OMPS2: <conf2|MPS>
        tstart("MPS")
        conf_end, OMPS2 = sampleMPS!(conf_end, mpsM, phiL_up[end], phiL_dn[end], latt)
        # Update phiR[1]
        phi_up, phi_dn = prodDetUpDn(conf_end)
        phiR_up[1] = expHk_half * phi_up
        phiR_dn[1] = expHk_half * phi_dn
        tend("MPS")
        #@assert abs(OMPS2-MPSOverlap(conf_end, mps)) < 1e-14    # Check MPS overlap



        # 4. Sample the auxiliary fields from right to left
        #    ODet: <conf1|BB...B|conf2>
        tstart("Det")
        for i=Ntau:-1:1
            # Sample the fields
            phi_up, phi_dn, ODet, auxflds[i] = sampleAuxField(phiL_up[i], phiL_dn[i], phiR_up[end-i], phiR_dn[end-i],
                                                              expV_up, expV_dn, auxflds[i], false)
            # Propagate B_K
            if i == 1
                phiR_up[end-i+1] = expHk_half * phi_up
                phiR_dn[end-i+1] = expHk_half * phi_dn
            else
                phiR_up[end-i+1] = expHk * phi_up
                phiR_dn[end-i+1] = expHk * phi_dn
            end

            # Measure at the center slice
            if (i == c+1)
                O = ODet * conj(OMPS1) * OMPS2
                phiLc_up = expHk_half_inv * phiL_up[i]
                phiLc_dn = expHk_half_inv * phiL_dn[i]
                phiRc_up = expHk_half * phi_up
                phiRc_dn = expHk_half * phi_dn
                measure!(phiLc_up, phiLc_dn, phiRc_up, phiRc_dn, sign(O), obs, para)
            end
        end
        tend("Det")


        # Write the observables
        if iMC%write_step == 0
            println(nsteps,": ",iMC,"/",N_samples)
            Eki = getObs(obs, "Ek")
            EVi = getObs(obs, "EV")
            Ei = getObs(obs, "E")
            nupi = getObs(obs, "nup")
            ndni = getObs(obs, "ndn")
            si = getObs(obs, "sign")

            println(file,iMC," ",Eki," ",EVi," ",Ei," ",si," ",nupi," ",ndni)
            flush(file)

            cleanObs!(obs)
        end
    end

    close(file)
    println("Total time: ")
    display(timer)
end

function main()
    Lx=2
    Ly=2
    tx=ty=1.0
    xpbc=false
    ypbc=false
    Nup = 2
    Ndn = 2
    U = 12.
    dtau = 0.02
    N_samples = 100000
    write_step = 100

    # Make H MPO
    N = Lx*Ly
    sites = siteinds("Electron", N, conserve_qns=true)
    ampo = Hubbard(Lx, Ly, tx, ty, U, 0, 0, 0, 0, xpbc, ypbc)
    H = MPO(ampo,sites)

    dir = "test"#ARGS[1]

    # Initialize MPS
    #states = RandomState(N; Nup, Ndn)
    states = ["Up","Dn","Dn","Up"]
    psi_init = MPS(sites, states)
    #en_init, psi_init = dmrg(H, psi_init; nsweeps=12, maxdim=[20,20,20,20,40,40,40,40,80,80,80,80], cutoff=[1e-14])
    en_init, psi_init = dmrg(H, psi_init; nsweeps=4, maxdim=[2,2,2,2], cutoff=[1e-14])
    init_D = maximum([linkdim(psi_init, i) for i in 1:N-1])
    println("Initial energy = ",en_init)

    # Write initial MPS to file
    writeMPS(psi_init,"initMPS.txt")
    error("stop")

    # Get exact energy from DMRG
    dims = [80,80,80,80,160,160,160,160,320,320,320,320,640]
    #dims = [4,8,16,16,16,16]
    E_GS, psi_GS = dmrg(H, psi_init; nsweeps=length(dims), maxdim=dims, cutoff=[1e-14])
    GS_D = maximum([linkdim(psi_GS, i) for i in 1:N-1])
    

    
    if false#folder_has_files(dir)
        println("$dir already has files. Do you want to continue?")
        ans = readline()
        if ans != "y"
            error("stopped")
        end
    end
    #dir = "data$(Lx)x$(Ly)_N$(Nup+Ndn)_dtau0.01/"
    # Measure the initial state and the ground state
    nups_init = expect(psi_init,"Nup")
    ndns_init = expect(psi_init,"Ndn")
    nups_GS = expect(psi_GS,"Nup")
    ndns_GS = expect(psi_GS,"Ndn")
    Ek_init, EV_init = getEkEV(psi_init, Lx, Ly, tx, ty, U, xpbc, ypbc)
    Ek_GS, EV_GS = getEkEV(psi_GS, Lx, Ly, tx, ty, U, xpbc, ypbc)
    # Write the information for the initial state and the ground state
    open(dir*"/init.dat","a") do file
        println(file,"randomSeed ",seed)
        println(file,"Lx ",Lx)
        println(file,"Ly ",Ly)
        println(file,"tx ",tx)
        println(file,"ty ",ty)
        println(file,"xpbc ",xpbc)
        println(file,"ypbc ",ypbc)
        println(file,"Nup ",Nup)
        println(file,"Ndn ",Ndn)
        println(file,"U ",U)
        println(file,"dtau ",dtau)
        println(file,"N_samples ",N_samples)
        println(file,"write_step ",write_step)
        println(file,"E0 ",en_init)
        println(file,"Ek0 ",Ek_init)
        println(file,"EV0 ",EV_init)
        println(file,"E_GS ",E_GS)
        println(file,"Ek_GS ",Ek_GS)
        println(file,"EV_GS ",EV_GS)
        println(file,"nup0 ",nups_init)
        println(file,"ndn0 ",ndns_init)
        println(file,"nup_GS ",nups_GS)
        println(file,"ndn_GS ",ndns_GS)
        println(file,"GS_MPS_D ",GS_D)
        println(file,"Init_MPS_D ",init_D)
        println(file,"Init_MPS_conf ",states)
    end

    for nsteps in [1]
        run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi_init, write_step, dir)
    end
end

main()
