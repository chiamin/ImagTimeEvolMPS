include("H_k.jl")
include("HSTrans.jl")
include("DetTools.jl")
include("Measure.jl")
using PyPlot
using LinearAlgebra

function initQMC(Lx::Int, Ly::Int, Lz::Int, tx::Float64,ty::Float64, tz::Float64, U::Float64, xpbc::Bool, ypbc::Bool, zpbc::Bool, dtau::Float64, nsteps::Int, Nsites::Int
)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Vector{Int}}}
    Hk = H_K(Lx,Ly,Lz,tx,ty,tz,xpbc,ypbc,zpbc)

    gamma = acosh(exp(0.5*dtau*U))
    expV_up = [exp(gamma),exp(-gamma)]
    expV_dn = [exp(-gamma),exp(gamma)]

    expHk_half = exp(-0.5*dtau*Hk)

    auxflds = []
    for i=1:2*nsteps
        push!(auxflds, ones(Int,Nsites))
    end

    return Hk, expHk_half, expV_up, expV_dn, auxflds
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
# If center at the very left: phis = <phi_beg|, B|phi_end>, BB|phi_end>, |phi_end>
# If center at the very right: phis = <phi_beg|, <phi_beg|BB, <phi_beg|B, |phi_end>
function initPhis(phi_beg::Matrix{T}, phi_end::Matrix{T}, expHk_half::Matrix{Float64}, auxflds::Vector{Vector{Int}}, expV::Vector{Float64})::Dict{Int,Matrix{T}} where T
    N = length(auxflds)
    @assert N >= 2

    phis = Dict{Int,Matrix{T}}()

    # Last one
    phis[N+1] = phi_end

    # Middle ones
    phi = copy(phi_end)
    for i=N:-1:1
        phis[i] = applyExpH(phis[i+1], auxflds[i], expHk_half, expV)
    end

    # First one
    phis[0] = phi_beg

    @assert length(phis) == N+2

    return phis
end

function overlap2(phi1_up::Matrix{T}, phi1_dn::Matrix{T}, phi2_up::Matrix{T}, phi2_dn::Matrix{T}, auxf::Vector{Int}, expV_up::Vector{Float64}, expV_dn::Vector{Float64}) where T
    phi_up = copy(phi1_up)
    phi_dn = copy(phi1_dn)
    applyV!(phi_up, auxf, expV_up)
    applyV!(phi_dn, auxf, expV_dn)
    O1 = overlap(phi_up, phi2_up)
    O2 = overlap(phi_dn, phi2_dn)
    return O1*O2
end

function sampleAuxField(
phi1_up::Matrix{T}, 
phi1_dn::Matrix{T}, 
phi2_up::Matrix{T}, 
phi2_dn::Matrix{T}, 
auxfld::Vector{Int64}, 
expHk_half::Matrix{Float64},
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64}; 
toRight::Bool
)::Tuple{Matrix{T},Matrix{T},Vector{Int64},Float64} where T
    @assert size(phi1_up) == size(phi1_dn) == size(phi2_up) == size(phi2_dn)

    Nsites,Npar = size(phi1_up)
    @assert length(auxfld) == Nsites

    function flip_field(field, i)
        field = copy(field)
        if field[i] == 1
            field[i] = 2
        elseif field[i] == 2
            field[i] = 1
        else
            throw(ErrorException("field must be either 1 or 2"))
        end
        return field
    end

    phi1_up = expHk_half * phi1_up
    phi1_dn = expHk_half * phi1_dn
    phi2_up = expHk_half * phi2_up
    phi2_dn = expHk_half * phi2_dn
    auxfld = copy(auxfld)
    O = overlap2(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld, expV_up, expV_dn)
    for i=1:Nsites
        auxfld_new = flip_field(auxfld, i)
        O_new = overlap2(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld_new, expV_up, expV_dn)

        P_new = abs(O_new) / (abs(O) + abs(O_new))
        if rand() < P_new
            O = O_new
            auxfld = auxfld_new
        end
    end

    if toRight
        phi_up = copy(phi1_up)
        phi_dn = copy(phi1_dn)
        applyV!(phi_up, auxfld, expV_up)
        applyV!(phi_dn, auxfld, expV_dn)
    else
        phi_up = copy(phi2_up)
        phi_dn = copy(phi2_dn)
        applyV!(phi_up, auxfld, expV_up)
        applyV!(phi_dn, auxfld, expV_dn)
    end
    phi_up = expHk_half * phi_up
    phi_dn = expHk_half * phi_dn
    return phi_up, phi_dn, auxfld, O
end

# Update x_1, ..., x_N with the probability
# p = <MPS|i><i|B(x_N)...B(x_1)|j><j|MPS>
# Here |i> and |j> are the first and the last state in phis,
# so they will not be updated in this function
function sampleAuxField_sweep_old!(
phis_up::Vector{Matrix{T}}, 
phis_dn::Vector{Matrix{T}}, 
auxflds::Vector{Vector{Int}}, 
expHk_half::Matrix{Float64}, 
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64}, 
obs=DefaultDict(0.)::DefaultDict;
toRight::Bool
) where T
    nsteps = length(auxflds)
    @assert nsteps >= 2
    @assert length(phis_up) == length(phis_dn) == nsteps+2

    # Left to right
    O = 0.
    if toRight
        for i=2:nsteps+1
            phi_up, phi_dn, auxflds[i-1], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i-1], expHk_half, expV_up, expV_dn; toRight=true)

            # Measure between phi, phis[i+1]
            if (i-1 == div(nsteps,2)) && (length(obs) != 0)
                measure!(phi_up, phi_dn, phis_up[i+1], phis_dn[i+1], O, obs)
            end

            cc = applyExpH(phis_up[i-1], auxflds[i-1], expHk_half, expV_up)
            println("kkk ",norm(cc-phi_up))

            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    # Right to left
    else
        for i=nsteps+1:-1:2
            phi_up, phi_dn, auxflds[i-1], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i-1], expHk_half, expV_up, expV_dn; toRight=false)

            # Measure between phis[i], phi
            if (i-1 == div(nsteps,2)+1) && (length(obs) != 0)
                measure!(phis_up[i-1], phis_dn[i-1], phi_up, phi_dn, O, obs)
            end

            cc = applyExpH(phis_up[i+1], auxflds[i-1], expHk_half, expV_up)
            println("ggg ",norm(cc-phi_up))

            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    end
    return O
end

# Update x_1, ..., x_N with the probability
# p = <MPS|i><i|B(x_N)...B(x_1)|j><j|MPS>
# Here |i> and |j> are the first and the last state in phis,
# so they will not be updated in this function
function sampleAuxField_sweep!(
phis_up::Dict{Int,Matrix{T}}, 
phis_dn::Dict{Int,Matrix{T}}, 
auxflds::Vector{Vector{Int}}, 
expHk_half::Matrix{Float64}, 
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64}, 
obs=DefaultDict(0.)::DefaultDict;
toRight::Bool
) where T
    N = length(auxflds)
    @assert N >= 2
    @assert length(phis_up) == length(phis_dn) == N+2

    # Left to right
    O = 0.
    if toRight
        for i=1:N
            phi_up, phi_dn, auxflds[i], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i], expHk_half, expV_up, expV_dn; toRight=true)

            # Measure between phi, phis[i+1]
            if (length(obs) != 0)# && (i == div(N,2))
                measure!(phi_up, phi_dn, phis_up[i+1], phis_dn[i+1], O, obs)
            end

            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    # Right to left
    else
        for i=N:-1:1
            phi_up, phi_dn, auxflds[i], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i], expHk_half, expV_up, expV_dn; toRight=false)

            # Measure between phis[i], phi
            if (length(obs) != 0)# && (i-1 == div(N,2))
                measure!(phis_up[i-1], phis_dn[i-1], phi_up, phi_dn, O, obs)
            end

            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    end
    return O
end


function applyExpH(
phi::Matrix{T}, 
auxfld::Vector{Int}, 
expHk_half::Matrix{Float64}, 
expV::Vector{Float64}, 
) where T
    Bphi = expHk_half * phi
    applyV!(Bphi, auxfld, expV)
    Bphi = expHk_half * Bphi
    return Bphi
end

