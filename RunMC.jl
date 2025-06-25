
function runMonteCarlo_MPS_MPS(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, mps, write_step, dir; suffix="")
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

    file = open(dir*"/ntau"*string(nsteps)*suffix*".dat","w")
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

function runMonteCarlo_Det_MPS(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, mps, phiT_up, phiT_dn, write_step, dir; suffix="")
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

    fileC = open(dir*"/c_ntau"*string(nsteps)*suffix*".dat","w")
    file1 = open(dir*"/l_ntau"*string(nsteps)*suffix*".dat","w")
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

function runMonteCarlo_Det_Det(Lx, Ly, tx, ty, xpbc, ypbc, Nup, Ndn, U, dtau, nsteps, N_samples, phiTL_up, phiTL_dn, phiTR_up, phiTR_dn, write_step, dir; suffix="")
    Nsites = Lx*Ly
    Npar = Nup+Ndn
    tau = dtau * nsteps

    # Initialize for QMC
    Hk, expV_up, expV_dn, auxflds = initQMC(Lx, Ly, tx, ty, U, xpbc, ypbc, dtau, nsteps, Nsites)
    expHk = exp(-dtau*Hk)
    expHk_half = exp(-0.5*dtau*Hk)
    expHk_half_inv = exp(+0.5*dtau*Hk)
    Ntau = length(auxflds)

    # Initialize phiT
    phi1_up, phi1_dn = phiTL_up, phiTL_dn
    phi2_up, phi2_dn = phiTR_up, phiTR_dn

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

    # Reset the timer
    treset()

    fileC = open(dir*"/c_ntau"*string(nsteps)*suffix*".dat","w")
    file1 = open(dir*"/l_ntau"*string(nsteps)*suffix*".dat","w")
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
                phiLc_up = expHk_half * phi_up
                phiLc_dn = expHk_half * phi_dn
                phiRc_up = expHk_half_inv * phiR_up[end-i]
                phiRc_dn = expHk_half_inv * phiR_dn[end-i]
                measure!(phiLc_up, phiLc_dn, phiRc_up, phiRc_dn, sign(ODet), obsC, para)
            end
        end
        tend("Det")

        # 2. Sample the auxiliary fields from right to left
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
                phiLc_up = expHk_half_inv * phiL_up[i]
                phiLc_dn = expHk_half_inv * phiL_dn[i]
                phiRc_up = expHk_half * phi_up
                phiRc_dn = expHk_half * phi_dn
                measure!(phiLc_up, phiLc_dn, phiRc_up, phiRc_dn, sign(ODet), obsC, para)
            end
            # Measure at the first slice
            if (i == 1)
                measure!(phiL_up[1], phiL_dn[1], phiR_up[end], phiR_dn[end], sign(ODet), obs1, para)
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

