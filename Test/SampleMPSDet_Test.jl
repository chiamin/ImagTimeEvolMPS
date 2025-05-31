using ITensors
using Test
using Random
using PyPlot
include("../DetTools.jl")
include("../EDTwoParticles.jl")
include("../SampleMPSDet.jl")
include("../ProdMPS.jl")

function Expansion_values4(mps, phi_up, phi_dn, t, U, pbc)
    L = length(mps)
    Hk = H_K(L,1,1,t,t,t,pbc,false,false)

    all_confs = []
    all_ws = []
    all_eks = []
    all_evs = []
    for i1=1:4
        for i2=1:4
            for i3=1:4
                for i4=1:4
                    conf = [i1,i2,i3,i4]
                    Nup, Ndn = getParNums(conf)
                    if Nup == 1 && Ndn == 1
                        push!(all_confs, conf)

                        w = getProbWeight(conf, mps, phi_up, phi_dn)    # <mps|i><i|phi>
                        push!(all_ws, w)

                        prod_up, prod_dn = prodDetUpDn(conf)
                        G_up = Greens_function(prod_up, phi_up)
                        G_dn = Greens_function(prod_dn, phi_dn)
                        Ek = kinetic_energy(G_up, G_dn, Hk)             # <i|H_k|phi> / <i|phi>
                        EV = potential_energy(G_up, G_dn, U)            # <i|H_V|phi> / <i|phi>
                        push!(all_eks, Ek)
                        push!(all_evs, EV)
                    end
                end
            end
        end
    end
    return all_confs, all_ws, all_eks, all_evs
end

function Expansion_values2(mps, phi_up, phi_dn, t, U, pbc)
    L = length(mps)
    Hk = H_K(L,1,1,t,t,t,pbc,false,false)

    all_confs = []
    all_ws = []
    all_eks = []
    all_evs = []
    for i1=1:4
        for i2=1:4
                    conf = [i1,i2]
                    Nup, Ndn = getParNums(conf)
                    if Nup == 1 && Ndn == 1
                        push!(all_confs, conf)

                        w = getProbWeight(conf, mps, phi_up, phi_dn)    # <mps|i><i|phi>
                        push!(all_ws, w)

                        prod_up, prod_dn = prodDetUpDn(conf)
                        G_up = Greens_function(prod_up, phi_up)
                        G_dn = Greens_function(prod_dn, phi_dn)
                        Ek = kinetic_energy(G_up, G_dn, Hk)             # <i|H_k|phi> / <i|phi>
                        EV = potential_energy(G_up, G_dn, U)            # <i|H_V|phi> / <i|phi>
                        push!(all_eks, Ek)
                        push!(all_evs, EV)
                    end
        end
    end
    return all_confs, all_ws, all_eks, all_evs
end

# Test <MPS|phi> = sum_i <MPS|i><i|phi>, 
# where |phi> is a Slatter determinant and |i> is a product state
function MPSDetOverlap_Test()
    Random.seed!(51321234)

    L = 2
    pbc = false
    t = 1.
    U = 8.

    phi_up = randomDet(L, 1)
    phi_dn = randomDet(L, 1)
    psi = MPS_GS(L, pbc, t, U)

    all_confs, all_ws, all_eks, all_evs = Expansion_values2(psi, phi_up, phi_dn, t, U, pbc)

    Ek_ED, EV_ED, O_ED = ED_measure(L, pbc, t, U, phi_up, phi_dn)

    # Check <psi|phi> == sum_i <psi|i><i|phi>
    Z = sum(all_ws) # sum_i <psi|i><i|phi>
    println("Z = ",O_ED," ",Z)
    @test abs(O_ED - Z) < 1e-13
    # Check Ek
    Ek_check = sum(all_ws .* all_eks) / Z
    println("Ek = ",Ek_ED," ",Ek_check)
    @test abs(Ek_ED-Ek_check) < 1e-13
    # Check EV
    EV_check = sum(all_ws .* all_evs) / Z
    println("EV = ",EV_ED," ",EV_check)
    @test abs(EV_ED-EV_check) < 1e-13
end

# 4-site, Nup=1 adn Ndn=1 
function Sampling_Test()
    Random.seed!(51321234)

    L = 2
    pbc = false
    t = 1.
    U = 8.

    phi_up = randomDet(L, 1)
    phi_dn = randomDet(L, 1)
    psi = MPS_GS(L, pbc, t, U)
    ppsi = makeProdMPS(psi)

    conf = ones(Int,L)
    conf[1] = 2
    conf[2] = 3

    latt = makeSquareLattice(L, 1, pbc, false)

    Hk = H_K(L,1,1,t,t,t,pbc,false,false)


    # Expansion
    all_confs, all_ws, all_eks, all_evs = Expansion_values2(psi, phi_up, phi_dn, t, U, pbc)
    all_abs_ws = abs.(all_ws)
    Zp = sum(all_abs_ws)

    N = length(all_confs)
    ws = fill(0.,N)

    # Sampling
    obs = makeObsDict()
    Nsamples = 20000
    for i=1:Nsamples
        for bond in latt.bonds
            conf, w = sampleBond(conf, ppsi, phi_up, phi_dn, bond)

            # Measure
            if i > 1000
                prod_up, prod_dn = prodDetUpDn(conf)
                G_up = Greens_function(prod_up, phi_up)
                G_dn = Greens_function(prod_dn, phi_dn)
                Ek = kinetic_energy(G_up, G_dn, Hk)
                EV = potential_energy(G_up, G_dn, U)
                sign_w = sign(w)

                measure!(obs["Ek"], Ek*sign_w)
                measure!(obs["EV"], EV*sign_w)
                measure!(obs["sign"], sign_w)

                ind = findfirst(x -> x == conf, all_confs)
                ws[ind] += 1.
            end
        end
        if i % 100 == 0
            Eki = getObs(obs["Ek"])
            EVi = getObs(obs["EV"])
            si = getObs(obs["sign"])
            println(i, " Ek:",Eki/si, " EV:",EVi/si, " sign:",si)
        end
    end

    # Check observables
    Eki = getObs(obs["Ek"])
    EVi = getObs(obs["EV"])
    si = getObs(obs["sign"])
    # 1. check sign
    all_signs = sign.(all_ws)
    sign_check = sum(all_abs_ws .* all_signs) / Zp
    println("sign: ",sign_check," ",si)
    # 2. check energy
    Ek_ED, EV_ED, O_ED = ED_measure(L, pbc, t, U, phi_up, phi_dn)
    println("Ek: ",Ek_ED," ",Eki/si)
    println("EV: ",EV_ED," ",EVi/si)

    # Plot probabilities
    # 1. from expansion
    p = PyPlot.plot(1:N, all_abs_ws./Zp)
    # 2. from sampling
    z = sum(ws)
    p = PyPlot.plot(1:N, ws/z)
    show()
end

MPSDetOverlap_Test()
Sampling_Test()
