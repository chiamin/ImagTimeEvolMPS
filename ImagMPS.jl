import Random
using LinearAlgebra
include("HSTrans.jl")
include("DetTools.jl")
include("SampleDet.jl")
include("Hamiltonian.jl")
include("dmrg.jl")
include("SampleMPSDet.jl")
include("Initial.jl")
include("Timer.jl")
using ITensorMPS
include("help.jl")

seed = 1234567890123
Random.seed!(seed)

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

function run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, mps, write_step)
    Nsites = Lx*Ly
    Npar = Nup+Ndn
    tau = dtau * nsteps

    # Initialize for QMC
    Hk, expHk_half, expV_up, expV_dn, auxflds = initQMC(Lx, Ly, tx, ty, U, xpbc, ypbc, dtau, nsteps, Nsites)
    Ntau = length(auxflds)

    # Initialize product states by sampling the MPS
    conf_beg = ITensorMPS.sample(mps)
    conf_end = deepcopy(conf_beg)
    println("Initial conf: ",conf_beg," ",conf_end)
    phi1_up, phi1_dn = prodDetUpDn(conf_beg)
    phi2_up, phi2_dn = prodDetUpDn(conf_end)
    # Compute the overlaps
    OMPS1 = MPSOverlap(conf_beg, mps)
    OMPS2 = MPSOverlap(conf_end, mps)

    # Initialize all the determinants
    phis_up = initPhis(phi1_up, phi2_up, expHk_half, auxflds, expV_up)
    phis_dn = initPhis(phi1_dn, phi2_dn, expHk_half, auxflds, expV_dn)

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

    dir = "data/"
    file = open(dir*"/ntau"*string(nsteps)*".dat","w")
    # Write the observables' names
    println(file,"step Ek EV sign nup ndn")


    # ** Check the determinant
    phis_up[1] = reOrthoDet(phis_up[1])
    phis_dn[1] = reOrthoDet(phis_dn[1])
    confs, ampls = get_normalized_slater_amplitudes(phis_up[1], phis_dn[1])
    EE = getEnergy(phi2_up, phi2_dn, Hk, U)
    println("EE: ",EE)
    EE = getEnergy(phis_up[1], phis_dn[1], Hk, U)
    println("EE: ",EE)

    # ** Check all the probability weights
    confs = generate_product_configurations(Nsites, Nup, Ndn)
    ws = Vector{Float64}()
    for conf in confs
        o1 = detProdOverlap(phis_up[1], phis_dn[1], conf)
        o2 = MPSOverlap(conf, mps)
        w = o1*o2
        if abs(w) < 1e-14
            w = 0.
        end
        push!(ws, w)
    end
    ws = ws ./ sum(ws)
    open("data/w.txt","w") do file
        for i=1:length(ws)
            println(file,confs[i]," ",ws[i])
        end
    end

    # Monte Carlo sampling
    x_all = generate_all_fields(Nsites)
    hist_conf1 = Dict(k => 0 for k in confs)
    hist_conf2 = Dict(k => 0 for k in confs)
    hist_x1 = Dict(x => 0 for x in x_all)
    hist_x2 = Dict(x => 0 for x in x_all)
    latt = makeSquareLattice(Lx, Ly, xpbc, ypbc)
    c = div(Ntau,2)
    for iMC=1:N_samples
        # 1. Sample the left product state
        #    OMPS1: <MPS|conf1>
        tstart("MPS")
        conf_beg, OMPS1 = sampleMPS!(conf_beg, mpsM, phis_up[1], phis_dn[1], latt)
        phis_up[0], phis_dn[0] = prodDetUpDn(conf_beg)
        tend("MPS")
        @assert abs(OMPS1-MPSOverlap(conf_beg, mps)) < 1e-14    # Check MPS overlap

        #hist[conf_beg] += 1


        # 2. Sample the auxiliary fields from left to right
        #    ODet: <conf1|BB...B|conf2>
        tstart("Det")
        for i=1:Ntau
            # Sample the fields
            phi_up, phi_dn, auxflds[i], ODet = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1],
                                                              auxflds[i], expHk_half, expV_up, expV_dn; toRight=true)
            # Update determinants
            phis_up[i] = phi_up
            phis_dn[i] = phi_dn

            # Measure at the center slice
            if (i == c)
                O = ODet * conj(OMPS1) * OMPS2
                measure!(phis_up[c], phis_dn[c], phis_up[c+1], phis_dn[c+1], sign(O), obs, para)


                if iMC == -1
                    G_up = Greens_function(phis_up[c], phis_up[c+1])
                    G_dn = Greens_function(phis_dn[c], phis_dn[c+1])
                    Ek = kinetic_energy(G_up, G_dn, Hk)
                    EV = potential_energy(G_up, G_dn, U)
                    odet = overlap(phis_up[c], phis_dn[c], phis_up[c+1], phis_dn[c+1])
                    println(conf_beg," ",conf_end," ",auxflds[1]," ",auxflds[2],": ",Ek+EV," ",Ek," ",EV,", ",ODet," ",OMPS1," ",OMPS2)
                    println(overlap(phis_up[c], phis_dn[c], phis_up[c], phis_dn[c]))
                    println(overlap(phis_up[c+1], phis_dn[c+1], phis_up[c+1], phis_dn[c+1]))
                    error("stop")
                end
            end
        end
        tend("Det")


        # 3. Sample the right product state
        #    OMPS2: <conf2|MPS>
        tstart("MPS")
        conf_end, OMPS2 = sampleMPS!(conf_end, mpsM, phis_up[Ntau], phis_dn[Ntau], latt)
        phis_up[Ntau+1], phis_dn[Ntau+1] = prodDetUpDn(conf_end)
        tend("MPS")
        @assert abs(OMPS2-MPSOverlap(conf_end, mps)) < 1e-14    # Check MPS overlap


        # 4. Sample the auxiliary fields from right to left
        #    ODet: <conf1|BB...B|conf2>
        tstart("Det")
        for i=Ntau:Ntau#-1:1
            # Sample the fields
            phi_up, phi_dn, auxflds[i], ODet = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1],
                                                              auxflds[i], expHk_half, expV_up, expV_dn; toRight=false)
            # Update determinants
            phis_up[i] = phi_up
            phis_dn[i] = phi_dn

            # Measure at the center slice
            if (i == c+1)
                O = ODet * conj(OMPS1) * OMPS2
                measure!(phis_up[c], phis_dn[c], phis_up[c+1], phis_dn[c+1], sign(O), obs, para)
            end
        end
        tend("Det")


        # Write the observables
        if iMC%write_step == 0
            println(nsteps,": ",iMC,"/",N_samples)
            Eki = getObs(obs, "Ek")
            EVi = getObs(obs, "EV")
            nupi = getObs(obs, "nup")
            ndni = getObs(obs, "ndn")
            si = getObs(obs, "sign")

            println(file,iMC," ",Eki," ",EVi," ",si," ",nupi," ",ndni)
            flush(file)

            cleanObs!(obs)
        end
    end

    #=open("data/hist.txt","w") do file
        for (key,val) in hist
            println(file,key," ",val)
        end
    end=#

    close(file)
    println("Total time: ")
    display(timer)
end

function measureMPS(psi)
    magz = expect(psi,"Sz")
    for (j,mz) in enumerate(magz)
        println("$j $mz")
    end
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
    dtau = 0.05
    #nsteps = 10
    N_samples = 10000
    write_step = 100

    # Initialize MPS
    en0, psi = Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps=10, maxdim=[10], cutoff=[1e-14])

    # Write psi to file
    confs, ampls = getAmplitudes(psi)
    open("data/psi0.txt","w") do file
        for i=1:length(confs)
            println(file,confs[i]," ",ampls[i])
        end
    end

    # Measure the initial MPS
    nups = expect(psi,"Nup")
    ndns = expect(psi,"Ndn")
    println("E0 = ",en0)

    # Get exact energy from DMRG
    en_DMRG, psi_DMRG = Hubbard_GS(Lx, Ly, tx, ty, U, xpbc, ypbc, Nup, Ndn; nsweeps=30, maxdim=[20,20,20,20,40,40,40,40,80,80,80,80,160], cutoff=[1e-14])
    Ek0, EV0 = getEkEV(psi, Lx, Ly, tx, ty, U, xpbc, ypbc)
    Ek_DMRG, EV_DMRG = getEkEV(psi_DMRG, Lx, Ly, tx, ty, U, xpbc, ypbc)

    # Write the information for the initial state
    open("data/init.dat","w") do file
        println(file,"E0 ",en0)
        println(file,"Ek0 ",Ek0)
        println(file,"EV0 ",EV0)
        println(file,"E_GS ",en_DMRG)
        println(file,"Ek_GS ",Ek_DMRG)
        println(file,"EV_GS ",EV_DMRG)
        println(file,"nup ",ndns)
        println(file,"ndn ",nups)
    end

    for nsteps in [1]#[10,20,30,40,50,60,70,80]
        run(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, psi, write_step)
    end
end

main()
