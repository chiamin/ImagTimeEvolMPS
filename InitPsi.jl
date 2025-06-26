using HDF5
import Random
using LinearAlgebra
using ITensorMPS
include("Hamiltonian.jl")
include("Initial.jl")
include("help.jl")

let
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
    dir = params["dir"]
    writeInitFile = params["writeInitFile"]
    mode = params["mode"]
    initDMRG_dims = params["initDMRG_dims"]
    GS_DMRG_dims = params["GS_DMRG_dims"]
    seed = params["randSeed"]
    suffix = get(params, "suffix", "")

    # Set random number seed
    if seed == 0
        seed = time_ns()
    end
    Random.seed!(seed)
    println("Random number seed: ",seed)

    # Make H MPO
    N = Lx*Ly
    sites = siteinds("Electron", N, conserve_qns=true)
    ampo = Hubbard(Lx, Ly, tx, ty, U, 0, 0, 0, 0, xpbc, ypbc)
    H = MPO(ampo,sites)

    # Initialize MPS
    if Lx == 2 && Ly == 2 && Nup == 2 && Ndn == 2
        states = ["Up","Dn","Dn","Up"]
    else
        states = RandomState(N; Nup, Ndn)
    end
    psi_init = MPS(sites, states)
    en_init, psi_init = dmrg(H, psi_init; nsweeps=length(initDMRG_dims), maxdim=initDMRG_dims, cutoff=[1e-14])
    init_D = maximum([linkdim(psi_init, i) for i in 1:N-1])
    println("Initial energy = ",en_init)

    # Write initial MPS to file
    #writeMPS(psi_init,dir*"/initMPS.txt")

    # Get the natural orbitals
    Cup = correlation_matrix(psi_init, "Cdagup", "Cup")
    Cdn = correlation_matrix(psi_init, "Cdagdn", "Cdn")
    eig_up = eigen(Cup)
    eig_dn = eigen(Cdn)
    phiT_up = eig_up.vectors[:,end-Nup+1:end]
    phiT_dn = eig_dn.vectors[:,end-Ndn+1:end]

    # Write to file
    #writeDet(phiT_up, phiT_dn, dir*"/initDet.txt")

    G_up = Greens_function(phiT_up, phiT_up)
    G_dn = Greens_function(phiT_dn, phiT_dn)
    H_k = Hk_onebody(Lx, Ly, tx, ty, 0.0, 0.0, 0.0, xpbc, ypbc)
    Ek_phiT = kinetic_energy(G_up, G_dn, H_k)
    EV_phiT = potential_energy(G_up, G_dn, U)
    E_phiT = Ek_phiT+EV_phiT


    # Measure the initial state and the ground state
    nups_init = expect(psi_init,"Nup")
    ndns_init = expect(psi_init,"Ndn")
    Ek_init, EV_init = getEkEV(psi_init, Lx, Ly, tx, ty, U, xpbc, ypbc)

    # Write the information for the initial state and the ground state
    open(dir*"/"*writeInitFile*"$suffix","a") do file
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
        println(file,"E0 ",en_init)
        println(file,"Ek0 ",Ek_init)
        println(file,"EV0 ",EV_init)
        println(file,"nup0 ",nups_init)
        println(file,"ndn0 ",ndns_init)
        println(file,"Init_MPS_conf ",states)
        println(file,"Init_MPS_D ",init_D)
        println(file,"E_phiT ",E_phiT)
        println(file,"Ek_phiT ",Ek_phiT)
        println(file,"EV_phiT ",EV_phiT)
        println(file,"suffix ",suffix)
    end

    f = h5open(dir*"/psi$suffix.h5","w")
    write(f,"psi_init",psi_init)
    write(f,"phiT_up",phiT_up)
    write(f,"phiT_dn",phiT_dn)
    close(f)
end
