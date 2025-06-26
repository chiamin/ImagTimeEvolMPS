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
    mode = params["mode"]
    seed = params["randSeed"]
    suffix = get(params, "suffix", "")
    InitPsi = params["InitPsi"]
    InfoFile = get(params, "InfoFile", "info")

    if seed == 0
        seed = time_ns()
    end
    Random.seed!(seed)

    open("$dir/$InfoFile$suffix", "w") do io
        for (k, v) in params
            println(io, "$k = $v")
        end
        println(io, "RandomNumberSeed = ",seed)
    end

    f = h5open(InitPsi,"r")
    psi_init = read(f,"psi_init",MPS)
    phiT_up = read(f["phiT_up"])
    phiT_dn = read(f["phiT_dn"])
    close(f)

    for (k, v) in params
        println("$k = $v")
    end

    if mode == "MPSMPS"
        runMonteCarlo_MPS_MPS(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi_init, write_step, dir; suffix=suffix)
    elseif mode == "DetMPS"
        runMonteCarlo_Det_MPS(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi_init, phiT_up, phiT_dn, write_step, dir; suffix=suffix)
    elseif mode == "DetDet"
        runMonteCarlo_Det_Det(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, phiT_up, phiT_dn, phiT_up, phiT_dn, write_step, dir; suffix=suffix)
    else
        error("Invalid mode: $mode")
    end
end

main()
