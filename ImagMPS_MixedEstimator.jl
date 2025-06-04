import Random
using LinearAlgebra
include("HSTrans.jl")
include("DetTools.jl")
include("SampleDet.jl")
include("Hamiltonian.jl")
include("SampleMPSDet.jl")
include("Initial.jl")
include("Timer.jl")
using ITensorMPS

seed = 196411470832444#time_ns()
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

function folder_has_files(path::String)::Bool
    if !isdir(path)
        error("The specified path is not a directory.")
    end
    # Filter out files only (not subdirectories)
    files = filter(f -> isfile(joinpath(path, f)), readdir(path))
    return !isempty(files)
end

function run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, mps, phiT_up, phiT_dn, write_step, dir)
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
    conf_end = ITensorMPS.sample(mps)
    phi1_up, phi1_dn = phiT_up, phiT_dn
    phi2_up, phi2_dn = prodDetUpDn(conf_end)

    # Compute the overlaps
    OMPS = MPSOverlap(conf_end, mps)

    println("Initial conf: ",conf_end)
    open(dir*"/init.dat","a") do file
        println(file,"Initial_conf: ",conf_end," ",OMPS)
    end

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
    obs1 = Dict{String,Any}()
    obsC = Dict{String,Any}()

    # Store some objects that will be used in measurement
    para = Dict{String,Any}()
    para["Hk"] = Hk
    para["U"] = U

    # Initialize MPS machine which is efficient in computing the overlap with a product state
    mpsM = makeProdMPS(mps)

    # Reset the timer
    treset()

    fileC = open(dir*"/C_ntau"*string(nsteps)*".dat","w")
    file1 = open(dir*"/1_ntau"*string(nsteps)*".dat","w")
    # Write the observables' names
    println(fileC,"step Ek EV E sign nup ndn")
    println(file1,"step Ek EV E sign nup ndn")



    # Monte Carlo sampling
    latt = makeSquareLattice(Lx, Ly, xpbc, ypbc)
    c = div(Ntau,2)
    for iMC=1:N_samples
        # 1. Sample the auxiliary fields from left to right
        #    ODet: <phiT|BB...B|conf_end>
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
                O = ODet * conj(OMPS)
                phiLc_up = expHk_half * phi_up
                phiLc_dn = expHk_half * phi_dn
                phiRc_up = expHk_half_inv * phiR_up[end-i]
                phiRc_dn = expHk_half_inv * phiR_dn[end-i]
                measure!(phiLc_up, phiLc_dn, phiRc_up, phiRc_dn, sign(O), obsC, para)
            end
            # Measure at the first slice
            if (i == 1)
                O = ODet * conj(OMPS)
                display(phiL_up[1])
                display(phiL_dn[1])
                println("----------------------------------------")
                measure!(phiL_up[1], phiL_dn[1], phiR_up[end], phiR_dn[end], sign(O), obs1, para)
                # Check energy
                phir = rand(size(phiT_up)...)
                G_up = Greens_function(phiL_up[1], phir)
                G_dn = Greens_function(phiL_dn[1], phir)
                Ek_phiT = kinetic_energy(G_up, G_dn, Hk)
                EV_phiT = potential_energy(G_up, G_dn, U)
                E_phiT = Ek_phiT+EV_phiT
                println("Check energy: ",E_phiT)
                println("overlap: ",ODet," ",conj(OMPS))
            end
        end
        tend("Det")


        # 2. Sample the right product state
        #    OMPS2: <conf2|MPS>
        tstart("MPS")
        conf_end, OMPS = sampleMPS!(conf_end, mpsM, phiL_up[end], phiL_dn[end], latt)
        # Update phiR[1]
        phi_up, phi_dn = prodDetUpDn(conf_end)
        phiR_up[1] = expHk_half * phi_up
        phiR_dn[1] = expHk_half * phi_dn
        tend("MPS")
        #@assert abs(OMPS2-MPSOverlap(conf_end, mps)) < 1e-14    # Check MPS overlap



        # 3. Sample the auxiliary fields from right to left
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
                O = ODet * conj(OMPS)
                phiLc_up = expHk_half_inv * phiL_up[i]
                phiLc_dn = expHk_half_inv * phiL_dn[i]
                phiRc_up = expHk_half * phi_up
                phiRc_dn = expHk_half * phi_dn
                measure!(phiLc_up, phiLc_dn, phiRc_up, phiRc_dn, sign(O), obsC, para)
            end
            # Measure at the first slice
            if (i == 1)
                O = ODet * conj(OMPS)
                measure!(phiL_up[1], phiL_dn[1], phiR_up[end], phiR_dn[end], sign(O), obs1, para)
            end
        end
        tend("Det")


        # Write the observables
        if iMC%write_step == 0
            println(nsteps,": ",iMC,"/",N_samples)
            Eki = getObs(obsC, "Ek")
            EVi = getObs(obsC, "EV")
            Ei = getObs(obsC, "E")
            nupi = getObs(obsC, "nup")
            ndni = getObs(obsC, "ndn")
            si = getObs(obsC, "sign")

            println(fileC,iMC," ",Eki," ",EVi," ",Ei," ",si," ",nupi," ",ndni)
            flush(fileC)

            Eki = getObs(obs1, "Ek")
            EVi = getObs(obs1, "EV")
            Ei = getObs(obs1, "E")
            nupi = getObs(obs1, "nup")
            ndni = getObs(obs1, "ndn")
            si = getObs(obs1, "sign")

            println(file1,iMC," ",Eki," ",EVi," ",Ei," ",si," ",nupi," ",ndni)
            flush(file1)

            cleanObs!(obsC)
        end
    end

    close(fileC)
    close(file1)
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
    U = 0.
    dtau = 1.0
    N_samples = 100
    write_step = 1


    # Make H MPO
    N = Lx*Ly
    sites = siteinds("Electron", N, conserve_qns=true)
    ampo = Hubbard(Lx, Ly, tx, ty, U, 0, 0, 0, 0, xpbc, ypbc)
    H = MPO(ampo,sites)

    # Initialize MPS
    states = RandomState(N; Nup, Ndn)
    psi_init = MPS(sites, states)
    en_init, psi_init = dmrg(H, psi_init; nsweeps=1, maxdim=[20,20,20,20,40,40,40,40,80,80,80,80], cutoff=[1e-14])
    init_D = maximum([linkdim(psi_init, i) for i in 1:N-1])
    println("Initial energy = ",en_init)

    # Write initial MPS to file
    #writeMPS(psi_init,"data5/initMPS.txt")

    # Get exact energy from DMRG
    dims = [80]#,80,80,80,160,160,160,160,320,320,320,320,640]
    E_GS, psi_GS = dmrg(H, psi_init; nsweeps=length(dims), maxdim=dims, cutoff=[1e-14])
    GS_D = maximum([linkdim(psi_GS, i) for i in 1:N-1])

    # Get the natural orbitals
    Cup = correlation_matrix(psi_init, "Cdagup", "Cup")
    Cdn = correlation_matrix(psi_init, "Cdagdn", "Cdn")
    eig_up = eigen(Cup)
    eig_dn = eigen(Cdn)
    phiT_up = eig_up.vectors[:,end-Nup+1:end]
    phiT_dn = eig_dn.vectors[:,end-Ndn+1:end]

    G_up = Greens_function(phiT_up, phiT_up)
    G_dn = Greens_function(phiT_dn, phiT_dn)
    H_k = Hk_onebody(Lx, Ly, tx, ty, 0.0, 0.0, 0.0, xpbc, ypbc)
    Ek_phiT = kinetic_energy(G_up, G_dn, H_k)
    EV_phiT = potential_energy(G_up, G_dn, U)
    E_phiT = Ek_phiT+EV_phiT


    # Exact
    eig = eigen(H_k)
    E_ex = sum(eig.values[1:Nup]) + sum(eig.values[1:Ndn])
    phiT_up = eig.vectors[:,1:Nup]
    phiT_dn = eig.vectors[:,1:Ndn]
    println("Exact energy: ",E_ex)
    # Check energy
    phir = rand(size(phiT_up)...)
    G_up = Greens_function(phiT_up, phir)
    G_dn = Greens_function(phiT_dn, phir)
    Ek_phiT = kinetic_energy(G_up, G_dn, H_k)
    EV_phiT = potential_energy(G_up, G_dn, U)
    E_phiT = Ek_phiT+EV_phiT
    println("Check energy: ",E_phiT)
    display(phiT_up)


    dir = "test"
    #=
    dir = "data_MixedEstimator/data4x4_N14_dtau0.02/10"
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
    open(dir*"/init.dat","w") do file
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
        println(file,"Init_MPS_conf ",states)
        println(file,"Init_MPS_D ",init_D)
        println(file,"GS_MPS_D ",GS_D)
        println(file,"E_phiT ",E_phiT)
        println(file,"Ek_phiT ",Ek_phiT)
        println(file,"EV_phiT ",EV_phiT)
    end
    =#

    for nsteps in [10]#,20,30,40,50]
        run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi_init, phiT_up, phiT_dn, write_step, dir)
    end
end

main()
