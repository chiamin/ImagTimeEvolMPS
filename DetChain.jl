include("HSTrans.jl")
include("DetTools.jl")
# Store the determinants appear in <phiL|B1.B2.B3..BN|phiR> in phis
#
# phiL and phiR are fixed
#
# There is a center location
# If center=3:
#             <phiL|B1.B2.B3..BN|phiR>
#                         ^
# <phis[1]| = <phiL|B1
# <phis[2]| = <phiL|B1.B2
# |phis[3]> =             B3...BN|phiR>
# |phis[N]> =                  BN|phiR>
mutable struct DetChain{T}
    phiL::Matrix{T}
    phiR::Matrix{T}
    phis::Vector{Matrix{T}}
    center::Int
end

# The initial center site is 1
function makeDetChain(phiL::Matrix{T}, phiR::Matrix{T}, expHk_half::Matrix{Float64}, auxflds::Vector{Vector{Int}}, expV::Vector{Float64})::DetChain{T} where T)
    N = length(auxflds)
    @assert N >= 2

    phis = Vector{Matrix{T}}(undef, N)

    phi = copy(phiL)
    for i=N:-1:1
        phis[i] = applyExpH(phis[i+1], auxflds[i], expHk_half, expV)
    end

    # Reorthogonalize all the determinants
    for (i,phi) in phis
        reOrthoDet!(phi)
    end

    return DetChain(phiL, phiR, phis, 1)
end

mutable struct DetChainUpDn{T}
    detsUp::DetChain{T}
    detsDn::DetChain{T}
end
