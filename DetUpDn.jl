include("HSTrans.jl")
include("DetTools.jl")
include("Measure.jl")

mutable struct DetUpDn{T}
    phi::Matrix{T}
    Nup::Int
end

function Base.:*(A::Matrix{T}, B::DetUpDn{T}) where T
    return A * B.phi
end

# expV[up/dn][x]
#=function applyV!(phi::DetUpDn{T}, aux_flds::Vector{Int64}, expV::Tuple{Vector{Float64},Vector{Float64}}) where T
    Nsites,Npar = size(phi)

    @assert size(aux_flds) == (Nsites,)

    for i=1:Nsites
        x = aux_flds[i]
        phi[i,:] *= expV[x]
    end
end=#
