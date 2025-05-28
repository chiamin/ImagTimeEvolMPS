include("HSTrans.jl")

function initQMC(Lx::Int, Ly::Int, tx::Float64, ty::Float64, U::Float64, xpbc::Bool, ypbc::Bool, dtau::Float64, nsteps::Int, Nsites::Int
)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Vector{Int}}}
    Hk = Hk_onebody(Lx, Ly, tx, ty, 0., 0., 0., xpbc, ypbc)

    gamma = acosh(exp(0.5*dtau*U))
    expV_up = [exp(gamma),exp(-gamma)]
    expV_dn = [exp(-gamma),exp(gamma)]

    auxflds = []
    for i=1:2*nsteps
        push!(auxflds, ones(Int,Nsites))
    end

    return Hk, expV_up, expV_dn, auxflds
end

# The center is at the left
# Suppose <phi_beg| B B |phi_end> = <phi_beg| (B_K/2 B_V B_K/2) (B_K/2 B_V B_K/2) |phi_end> 
# If center at the very left: phis = <phi_beg|, B_K/2 B_V B_K B_V B_K/2|phi_end>, B_K B_V B_K/2|phi_end>, B_K/2|phi_end>
# If center at the very right: phis = <phi_beg|B_K/2, <phi_beg|B_K/2 B_V B_K, <phi_beg|B_K/2 B_V B_K B_V B_K/2, |phi_end>
function initPhis(phi_beg::Matrix{T}, phi_end::Matrix{T}, expHk::Matrix{Float64}, expHk_half::Matrix{Float64}, auxflds::Vector{Vector{Int}}, expV::Vector{Float64})::Dict{Int,Matrix{T}} where T
    N = length(auxflds)
    @assert N >= 2

    phis = Dict{Int,Matrix{T}}()

    # Last one
    phis[N+1] = expHk_half * phi_end

    # Middle ones
    phi = copy(phi_end)
    for i=N:-1:1
        phis[i] = applyExpH(phis[i+1], auxflds[i], expHk_half, expV)
        phis[i] = reOrthoDet(phis[i])   # reorthogonalize determinant
    end

    # First one
    phis[0] = phi_beg

    @assert length(phis) == N+2

    return phis
end


# The center is at the left
# Suppose <phi_beg| B B |phi_end> 
# If center at the very left: phis = <phi_beg|, B|phi_end>, BB|phi_end>, |phi_end>
# If center at the very right: phis = <phi_beg|, <phi_beg|BB, <phi_beg|B, |phi_end>
function initPhis_old(phi_beg::Matrix{T}, phi_end::Matrix{T}, expHk_half::Matrix{Float64}, auxflds::Vector{Vector{Int}}, expV::Vector{Float64})::Vector{Matrix{T}} where T
    nsteps = length(auxflds)
    @assert nsteps >= 2

    phis = []

    # Last one
    push!(phis, phi_end)

    # Middle ones
    phi = copy(phi_end)
    for i=1:nsteps
        phi = expHk_half * phi
        applyV!(phi, auxflds[i], expV)
        phi = expHk_half * phi
        pushfirst!(phis, phi)
    end

    # First one
    pushfirst!(phis, phi_beg)

    @assert length(phis) == nsteps+2

    return phis
end

# The center is at the left
# Suppose <phi_beg| B B |phi_end> 
# If center at the very left: phis = <phi_beg|, BB|phi_end>, B|phi_end>, |phi_end>
# If center at the very right: phis = <phi_beg|, <phi_beg|B, <phi_beg|BB, |phi_end>
function initPhis_old2(phi_beg::Matrix{T}, phi_end::Matrix{T}, expHk_half::Matrix{Float64}, auxflds::Vector{Vector{Int}}, expV::Vector{Float64})::Dict{Int,Matrix{T}} where T
    N = length(auxflds)
    @assert N >= 2

    phis = Dict{Int,Matrix{T}}()

    # Last one
    phis[N+1] = phi_end

    # Middle ones
    phi = copy(phi_end)
    for i=N:-1:1
        phis[i] = applyExpH(phis[i+1], auxflds[i], expHk_half, expV)
        phis[i] = reOrthoDet(phis[i])   # reorthogonalize determinant
    end

    # First one
    phis[0] = phi_beg

    @assert length(phis) == N+2

    return phis
end
