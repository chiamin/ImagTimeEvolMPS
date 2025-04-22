include("H_k.jl")
include("HSTrans.jl")
include("DetTools.jl")
include("Measure.jl")
include("Initial.jl")
using PyPlot
using LinearAlgebra

# <phi|B_V|phi>
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

    # Apply B_K
    phi1_up = expHk_half * phi1_up
    phi1_dn = expHk_half * phi1_dn
    phi2_up = expHk_half * phi2_up
    phi2_dn = expHk_half * phi2_dn
    # ----  Apply B_V ----
    auxfld = copy(auxfld)
    O = overlap2(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld, expV_up, expV_dn)              # initial overlap
    # Sample the auxiliary fields at all sites
    for i=1:Nsites
        auxfld_new = flip_field(auxfld, i)                                                  # flip the field on one site
        O_new = overlap2(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld_new, expV_up, expV_dn)  # compute the new overlap
        P_new = abs(O_new) / (abs(O) + abs(O_new))                                          # compute the probability
        rrr = rand()
        if rrr < P_new                                                                   # sample the field
            O = O_new
            auxfld = auxfld_new
        end
    end
    # Apply B_V on phi1 or phi2 depending on the sweeping direction
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
    # --------------------
    # Apply B_K
    phi_up = expHk_half * phi_up
    phi_dn = expHk_half * phi_dn
    return phi_up, phi_dn, auxfld, O
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
obs::Dict{String,Any},
para::Dict{String,Any};
toRight::Bool
) where T
    N = length(auxflds)
    @assert N >= 2
    @assert length(phis_up) == length(phis_dn) == N+2

    # Left to right
    O = 0.
    if toRight
        for i=1:N
            phi_up, phi_dn, auxflds[i], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i],
                                                           expHk_half, expV_up, expV_dn; toRight=true)

            #O *= (Osign_up * Osign_dn)

            # Measure at the center, between phi, phis[i+1]
            if (i == div(N,2))
                measure!(phi_up, phi_dn, phis_up[i+1], phis_dn[i+1], sign(O), obs, para)
            end

            # Stabilize the determinant
            phi_up = reOrthoDet(phi_up)
            phi_dn = reOrthoDet(phi_dn)

            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    # Right to left
    else
        for i=N:-1:1
            phi_up, phi_dn, auxflds[i], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i],
                                                           expHk_half, expV_up, expV_dn; toRight=false)

            #O *= (Osign_up * Osign_dn)

            # Measure at the center, between phis[i], phi
            if (i-1 == div(N,2))
                measure!(phis_up[i-1], phis_dn[i-1], phi_up, phi_dn, sign(O), obs, para)
            end

            # Stabilize the determinant
            phi_up = reOrthoDet(phi_up)
            phi_dn = reOrthoDet(phi_dn)

            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    end
    return O
end

# ================================== debug code ========================================
function check_overlap(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld, auxfld_new, expV_up, expV_dn)
    p1up = reOrthoDet(phi1_up)
    p1dn = reOrthoDet(phi1_dn)
    p2up = reOrthoDet(phi2_up)
    p2dn = reOrthoDet(phi2_dn)
    O = overlap2(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld, expV_up, expV_dn)
    O_new = overlap2(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld_new, expV_up, expV_dn)
    O_orth = overlap2(p1up, p1dn, p2up, p2dn, auxfld, expV_up, expV_dn)
    O_orth_new = overlap2(p1up, p1dn, p2up, p2dn, auxfld_new, expV_up, expV_dn)
    # Check overlap
    #@assert abs(O - O_orth) > 1e-4
    #@assert abs(O_new - O_orth_new) > 1e-4
    @assert abs(O/O_new - O_orth/O_orth_new) < 1e-10
    # Check sign
    @assert sign(O) == sign(O_orth)
    @assert sign(O_new) == sign(O_orth_new)
    # Check probability
    P = abs(O_new) / (abs(O) + abs(O_new))
    P_orth = abs(O_orth_new) / (abs(O_orth) + abs(O_orth_new))
    @assert abs(P - P_orth) < 1e-12
end
