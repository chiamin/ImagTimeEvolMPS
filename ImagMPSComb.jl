using HDF5
import Random
using LinearAlgebra
using ITensorMPS
include("HSTrans.jl")
include("DetTools.jl")
include("SampleDet.jl")
include("Hamiltonian.jl")
include("SampleMPSDet.jl")
include("Initial.jl")
include("Timer.jl")
include("help.jl")
include("RunMC.jl")

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

function main()
    params = read_params(ARGS[1])
    # Update parameters with arguments
    args = parse_args(ARGS)
    override_params_with_args!(params, args)

    Lx = params["Lx"]
    Ly = params["Ly"]
    tx = params["tx"]
    ty = params["ty"]
    xpbc = params["xPBC"]
    ypbc = params["yPBC"]
    Nup = params["Nup"]
    Ndn = params["Ndn"]
    U = params["U"]
    dtau = params["dtau"]
    nsteps = params["nsteps"]
    N_samples = params["N_samples"]
    write_step = params["write_step"]
    dir = params["dir"]
    writeInitFile = params["writeInitFile"]
    mode = params["mode"]
    initDMRG_dims = params["initDMRG_dims"]
    GS_DMRG_dims = params["GS_DMRG_dims"]
    seed = params["randSeed"]
    init_mode = params["InitMode"]
    suffix = get(params, "suffix", "")

    if seed == 0
        seed = time_ns()
    end
    Random.seed!(seed)
    println("Random number seed: ",seed)

    if init_mode
        # Make H MPO
        N = Lx*Ly
        sites = siteinds("Electron", N, conserve_qns=true)
        ampo = Hubbard(Lx, Ly, tx, ty, U, 0, 0, 0, 0, xpbc, ypbc)
        H = MPO(ampo,sites)

        # Initialize MPS
        states = ["Up","Dn","Dn","Up"]#RandomState(N; Nup, Ndn)
        psi_init = MPS(sites, states)
        en_init, psi_init = dmrg(H, psi_init; nsweeps=length(initDMRG_dims), maxdim=initDMRG_dims, cutoff=[1e-14])
        init_D = maximum([linkdim(psi_init, i) for i in 1:N-1])
        println("Initial energy = ",en_init)

        # Write initial MPS to file
        writeMPS(psi_init,dir*"/initMPS.txt")

        # Get exact energy from DMRG
        E_GS, psi_GS = dmrg(H, psi_init; nsweeps=length(GS_DMRG_dims), maxdim=GS_DMRG_dims, cutoff=[1e-14])
        GS_D = maximum([linkdim(psi_GS, i) for i in 1:N-1])


        # Get the natural orbitals
        Cup = correlation_matrix(psi_init, "Cdagup", "Cup")
        Cdn = correlation_matrix(psi_init, "Cdagdn", "Cdn")
        eig_up = eigen(Cup)
        eig_dn = eigen(Cdn)
        phiT_up = eig_up.vectors[:,end-Nup+1:end]
        phiT_dn = eig_dn.vectors[:,end-Ndn+1:end]

        # Write to file
        writeDet(phiT_up, phiT_dn, dir*"/initDet.txt")

        G_up = Greens_function(phiT_up, phiT_up)
        G_dn = Greens_function(phiT_dn, phiT_dn)
        H_k = Hk_onebody(Lx, Ly, tx, ty, 0.0, 0.0, 0.0, xpbc, ypbc)
        Ek_phiT = kinetic_energy(G_up, G_dn, H_k)
        EV_phiT = potential_energy(G_up, G_dn, U)
        E_phiT = Ek_phiT+EV_phiT



        if false#folder_has_files(dir)
            println("$dir already has files. Do you want to continue?")
            ans = readline()
            if ans != "y"
                error("stopped")
            end
        end

        # Measure the initial state and the ground state
        nups_init = expect(psi_init,"Nup")
        ndns_init = expect(psi_init,"Ndn")
        nups_GS = expect(psi_GS,"Nup")
        ndns_GS = expect(psi_GS,"Ndn")
        Ek_init, EV_init = getEkEV(psi_init, Lx, Ly, tx, ty, U, xpbc, ypbc)
        Ek_GS, EV_GS = getEkEV(psi_GS, Lx, Ly, tx, ty, U, xpbc, ypbc)
        # Write the information for the initial state and the ground state
        open(dir*"/"*writeInitFile,"a") do file
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
            println(file,"nsteps ",nsteps)
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
            println(file,"suffix ",suffix)
        end

        f = h5open(dir*"/psi.h5","w")
        write(f,"psi_init",psi_init)
        write(f,"psi_GS",psi_GS)
        write(f,"phiT_up",phiT_up)
        write(f,"phiT_dn",phiT_dn)
        close(f)
    # Not initialze mode
    else
        f = h5open(dir*"/psi.h5","r")
        psi_init = read(f,"psi_init",MPS)
        phiT_up = read(f["phiT_up"])
        phiT_dn = read(f["phiT_dn"])
        close(f)

        for (k, v) in params
            println("$k = $v")
        end

        if mode == "MPS2"
            runMonteCarlo_MPS_MPS(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi_init, write_step, dir; suffix=suffix)
        elseif mode == "DetMPS"
            runMonteCarlo_Det_MPS(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi_init, phiT_up, phiT_dn, write_step, dir; suffix=suffix)
        elseif mode == "DetDet"
            runMonteCarlo_Det_Det(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, phiT_up, phiT_dn, phiT_up, phiT_dn, write_step, dir; suffix=suffix)
        end
    end
end

main()
