include("../SampleOcc.jl")
include("../SquareLattice.jl")
include("../SampleTools.jl")
using Test
using Statistics
using LinearAlgebra
using Plots

function check_particle_number_conservation()
    function getParNum(i)
        Nup,Ndn = 0,0
        if i == 2 || i == 4
            Nup += 1
        end
        if i == 3 || i == 4
            Ndn += 1
        end
        return Nup, Ndn
    end

    function getParNums(conf)
        Nup,Ndn = 0,0
        for i in conf
            nup, ndn = getParNum(i)
            Nup += nup
            Ndn += ndn
        end
        return Nup, Ndn
    end

    for i1=1:4
        for i2=1:4
            confs = AllConfs(i1,i2)
            Nup, Ndn = getParNums(confs[1])
            for j=2:length(confs)
                nup, ndn = getParNums(confs[j])
                @test Nup == nup
                @test Ndn == ndn
            end
        end
    end
end

function SampleBond_Test(Nsample)
    function probDist(conf)
        return sum(conf)
    end

    function getParNum(i)
        local Nup,Ndn = 0,0
        if i == 2 || i == 4
            Nup += 1
        end
        if i == 3 || i == 4
            Ndn += 1
        end
        return Nup, Ndn
    end
    function getParNums(conf)
        local Nup,Ndn = 0,0
        for i in conf
            nup, ndn = getParNum(i)
            Nup += nup
            Ndn += ndn
        end
        return Nup, Ndn
    end
    function getConfDict(Nup, Ndn)
        local indDict = Dict()
        local confs = []
        local ind = 1
        for i1=1:4
            local nup1,ndn1 = getParNum(i1)
            for i2=1:4
                local nup2,ndn2 = getParNum(i2)
                for i3=1:4
                    local nup3,ndn3 = getParNum(i3)
                    for i4=1:4
                        local nup4,ndn4 = getParNum(i4)
                        if Nup == nup1+nup2+nup3+nup4 && Ndn == ndn1+ndn2+ndn3+ndn4
                            indDict[(i1,i2,i3,i4)] = ind
                            push!(confs, [i1,i2,i3,i4])
                            ind += 1
                        end
                    end
                end
            end
        end
        return indDict, confs
    end

    # Initial configuration
    s = [1,2,3,1]
    Nup,Ndn = getParNums(s)
    L = length(s)

    confInd, confs = getConfDict(Nup,Ndn)

    P_dist = []
    for conf in confs
        push!(P_dist, probDist(conf))
    end
    P_dist = P_dist ./ sum(P_dist)

    Ek_sum, sign_sum = 0., 0.
    cc = 0.
    dist = zeros(length(confInd))
    for i=1:Nsample
        # For each bond
        for j=1:L-1
            # Store all the configurations
            s2all = AllConfs(s[j],s[j+1])
            sall = [copy(s)]
            for s2 in s2all[2:end]
                println(s2)


                s[j],s[j+1] = s2[1],s2[2]
                push!(sall, copy(s))

                nup, ndn = getParNums(s)
                @test nup == Nup
                @test ndn == Ndn
            end
            


            # Compute probabilities
            ps = Vector{Float64}()
            for si in sall
                p = probDist(si)
                push!(ps, p)
            end

            # Sample configuration
            ic = sampleProb(ps)
            s = sall[ic]
            p = ps[ic]

            # Measure
            ind = confInd[Tuple(s)]
            dist[ind] += 1.
        end
    end
    dist = dist ./ sum(dist)

    x = 1:length(confs)
    plot(x, P_dist)
    p = plot!(x, dist)
    display(p)
end

function SampleConf_Latt_Test(Nsample, tol)
    lx, ly = 4,4
    xpbc = ypbc = true
    latt = makeSquareLattice(lx, ly, xpbc, ypbc)

    N = lx*ly
    occs = [3, 5, 10, 14]

    dist = zeros(N)
    for i=1:Nsample
        occs, accepted = sampleOccs(latt, occs)
        for j in occs
            dist[j] += 1.
        end
    end
    dist /= mean(dist)
    @test norm(dist-ones(N)) < tol
end

function SampleConf_Test(Nsample, tol)
    lx, ly = 4,4
    xpbc = ypbc = true

    N = lx*ly
    occs = [3, 5, 10, 14]

    dist = zeros(N)
    for i=1:Nsample
        occs, accepted = sampleOccs(N, occs)
        for j in occs
            dist[j] += 1.
        end
    end
    dist /= mean(dist)
    @test norm(dist-ones(N)) < tol
end

SampleBond_Test(1000000)
check_particle_number_conservation()
SampleConf_Test(10000000, 1e-2)
SampleConf_Latt_Test(10000000, 1e-2)
