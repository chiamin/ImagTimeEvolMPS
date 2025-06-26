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

    # Get exact energy from DMRG
    E_GS, psi_GS = dmrg(H, psi_init; nsweeps=length(GS_DMRG_dims), maxdim=GS_DMRG_dims, cutoff=[1e-14])
    GS_D = maximum([linkdim(psi_GS, i) for i in 1:N-1])


    nups_GS = expect(psi_GS,"Nup")
    ndns_GS = expect(psi_GS,"Ndn")
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
        println(file,"E_GS ",E_GS)
        println(file,"Ek_GS ",Ek_GS)
        println(file,"EV_GS ",EV_GS)
        println(file,"nup_GS ",nups_GS)
        println(file,"ndn_GS ",ndns_GS)
        println(file,"GS_MPS_D ",GS_D)
        println(file,"suffix ",suffix)
    end

    f = h5open(dir*"/psiGS.h5","w")
    write(f,"psi_GS",psi_GS)
    close(f)
end
