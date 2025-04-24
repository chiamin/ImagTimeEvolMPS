using ITensors
using Test
using Random
using StatsBase
include("EDTwoParticles.jl")
include("Hamiltonian.jl")
include("DetProdOverlap.jl")
include("SquareLattice.jl")
include("SampleOcc.jl")
include("Measure.jl")
include("DetTools.jl")
include("ProdMPS.jl")

# Return the relative sign between MPS and determinant ordering for a configuration
# An MPS has the ordering: 1_up, 1_dn, 2_up, 2_dn, ....
# A determinant has ordering: 1_up, 2_up, ..., 1_dn, 2_dn, ...
# The sign is determined by the number of swaps from MPS ordering to determinant ordering
function relativeSign(conf::Vector{Int})
    up_sites, dn_sites = getOccupiedSites(conf)
    N_swap = 0  # The number of swapping needs to be applied
    for i_dn in dn_sites
        # Find the number of elements larger than i in up_sites
        for i_up in reverse(up_sites)
            if i_up > i_dn
                N_swap += 1
            else
                break
            end
        end
    end

    sign = 1
    if N_swap % 2 == 1
        sign = -1
    end
    return sign
end

function MPSOverlap(conf::Vector{Int}, mps::MPS)
    state = []
    for i in conf
        if i == 1
            push!(state,"Emp")
        elseif i == 2
            push!(state,"Up")
        elseif i == 3
            push!(state,"Dn")
        elseif i == 4
            push!(state,"UpDn")
        end
    end
    sites = siteinds(mps)
    psi = MPS(sites, state)
    o = inner(mps, psi)
    sign = relativeSign(conf)
    return o*sign
end

function MPSOverlap(conf::Vector{Int}, pmps::ProdMPS)
    o = getOverlap!(pmps, conf)
    sign = relativeSign(conf)
    return o*sign
end

# <mps|conf><conf|phi>, where
# |conf> is a product state, and
# |phi> = |phi_up> |phi_dn> is a determinant
function getProbWeight(conf::Vector{Int}, mps, phi_up::Matrix{T}, phi_dn::Matrix{T}) where T
    o1 = detProdOverlap(phi_up, phi_dn, conf)
    o2 = MPSOverlap(conf, mps)
    o = o1*o2
    return o
end

# Sample the product state |i> with probability <mps|i><i|phi>
# Can be further optimized
function sampleBond(conf::Vector{Int}, mps, phi_up::Matrix{T}, phi_dn::Matrix{T}, bond::Vector{Int}) where T
    i1, i2 = bond

    # Store all the configurations
    confs_2s = AllConfs(conf[i1], conf[i2])
    N = length(confs_2s)
    confs = Vector{Vector{Int}}()
    for (s1,s2) in confs_2s
        conf_i = copy(conf)
        conf_i[i1] = s1
        conf_i[i2] = s2

        push!(confs, conf_i)
    end

    # Compute the probability weights for all the configurations
    ws = Vector{Float64}()
    for conf_i in confs
        wi = getProbWeight(conf_i, mps, phi_up, phi_dn)
        push!(ws, wi)
    end

    # Random sampling
    ps = abs.(ws)
    ind = StatsBase.sample(Weights(ps))

    return confs[ind], ws[ind]
end

function sampleMPS!(conf::Vector{Int}, psi, phi_up::Matrix{T}, phi_dn::Matrix{T}, latt, obs=DefaultDict(0.)) where T
    O = 0.
    for bond in latt.bonds
        conf, O = sampleBond(conf, psi, phi_up, phi_dn, bond)

        # Measure
        if length(obs) != 0
            prod_up, prod_dn = prodDetUpDn(conf)
            measure!(prod_up, prod_dn, phi_up, phi_dn, w, obs)
        end
    end
    return conf, O
end
