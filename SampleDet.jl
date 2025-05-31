include("HSTrans.jl")
include("DetTools.jl")
include("Measure.jl")
include("Initial.jl")
#include("DetChain.jl")
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

#=
# Consider <phi'|B_V(x_i)|phi>, where x_i is the auxiliary field on site i
# Using Sherman-Morrison formula to get O(x_i)/O,
# where O=<phi'|phi> and O(x_i)=<phi'|B_V(x_i)|phi>
# Input: Oinv = O^-1
#function overlapBVxi(Oinv::Matrix{T}, 
#
# Sample x_i with probability <phi1|B_V(x_i)|phi2> on site i, and then update |phi2>
# phi1_up, phi1_dn, phi2_up, phi2_dn are one row in the determinant matrices, but with the shapes as column vectors
function sampleAuxFieldOneSite(
phi1_up::Vector{T}, 
phi1_dn::Vector{T}, 
phi2_up::Vector{T}, 
phi2_dn::Vector{T}, 
Oinv_up::Matrix{T}, 
Oinv_dn::Matrix{T},
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64}
)::Tuple{Int,Matrix{T},Matrix{T},Float64} where T
    # Use Sherman-Morrison formula to compute overlap ratio <phi1|B_V(x_site)|phi2> / <phi1|phi2>
    # O^-1 |phi1'(site)>
    R_up = Oinv_up * conj(phi1_up)
    R_dn = Oinv_dn * conj(phi1_dn)
    # <phi2(site)| O^-1
    L_up = transpose(phi2_up) * Oinv_up
    L_dn = transpose(phi2_dn) * Oinv_dn
    # <phi2(site)| O^-1 |phi1'(site)>
    Gii_up = transpose(phi2_up) * R_up
    Gii_dn = transpose(phi2_dn) * R_dn
    # (expV(x)-1) * <phi2(site)| O^-1 |phi1'(site)>
    Gx_up = [(expV_up[1]-1.0) * Gii_up, (expV_up[2]-1.0) * Gii_up]
    Gx_dn = [(expV_dn[1]-1.0) * Gii_dn, (expV_dn[2]-1.0) * Gii_dn]
    # <phi1|B_V(x_site)|phi2> / <phi1|phi2> = 1 + Gx_up[x]
    O1_up = 1.0 + Gx_up[1]
    O2_up = 1.0 + Gx_up[2]
    O1_dn = 1.0 + Gx_dn[1]
    O2_dn = 1.0 + Gx_dn[2]

    # <phi1|B_V(x)|phi2> / <phi1|phi2> = 1 + Gx_up[x]
    O1 = O1_up * O1_dn
    O2 = O2_up * O2_dn
    Ox = [O1, O2]

    # Sample field
    P1 = abs(O1)/(abs(O1)+abs(O2))
    println("* ",P1)
    x = 2
    rnd = rand()
    if rnd < P1
        x = 1
    end

    # Update O^-1 by Sherman-Morrison formula
    factor_up = (1.0 - expV_up[x]) / (1.0 + Gx_up[x])
    factor_dn = (1.0 - expV_dn[x]) / (1.0 + Gx_dn[x])
    Oinv_up = Oinv_up + factor_up * R_up * L_up
    Oinv_dn = Oinv_dn + factor_dn * R_dn * L_dn

    return x, Oinv_up, Oinv_dn, Ox[x]
end

# Sample x with probability <phi1|B_V(x)|phi2>
function sampleAuxField(
phi1_up::Matrix{T}, 
phi1_dn::Matrix{T}, 
phi2_up::Matrix{T}, 
phi2_dn::Matrix{T}, 
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64},
toRight::Bool
)::Tuple{Matrix{T},Matrix{T},Float64} where T
    @assert size(phi1_up) == size(phi1_dn) == size(phi2_up) == size(phi2_dn)

    Nsites,Npar = size(phi1_up)

    # Compute overlap matrices and inverse overlaps
    Oup = phi1_up' * phi2_up
    Odn = phi1_dn' * phi2_dn
    Oinv_up = inv(Oup)
    Oinv_dn = inv(Odn)
    O = det(Oup) * det(Odn)

    phi1_up = copy(phi1_up)
    phi1_dn = copy(phi1_dn)
    phi2_up = copy(phi2_up)
    phi2_dn = copy(phi2_dn)

    for i=1:Nsites
        # Sample xi
        x, Oinv_up, Oinv_dn, O_ratio = sampleAuxFieldOneSite(phi1_up[i,:], phi1_dn[i,:], phi2_up[i,:], phi2_dn[i,:], Oinv_up, Oinv_dn, expV_up, expV_dn)
        # Update overlap
        O *= O_ratio
        # Update phi
        if toRight
            # Propagate phi1
            phi1_up[i,:] .*= expV_up[x]
            phi1_dn[i,:] .*= expV_dn[x]
        else
            # Propagate phi2
            phi2_up[i,:] .*= expV_up[x]
            phi2_dn[i,:] .*= expV_dn[x]
        end
    end

    # Stabilize the determinant
    if toRight
        #phi_up = reOrthoDet(phi1_up)
        #phi_dn = reOrthoDet(phi1_dn)
        phi_up = phi1_up
        phi_dn = phi1_dn
    else
        #phi_up = reOrthoDet(phi2_up)
        #phi_dn = reOrthoDet(phi2_dn)
        phi_up = phi2_up
        phi_dn = phi2_dn
    end

    return phi_up, phi_dn, O
end
=#

# Use Sherman-Morrison formula to compute the overlap and inverse overlap matrix
# Given Oinv = inv(Phi1' * Phi2), and O = <phi1|phi2>
# Consider O(x_i) = <phi1|B_V(x_i)|phi2>, where B_V(x_i)*Phi2 mutiplies a factor b to the ith row of Phi2
# Compute the overlap ratio O(x_i)/O and the inverse overlap inv(Phi1' * B_V(x_i) * Phi2)
#
# phi1 and phi2 are one row in the determinant matrices, but with the shapes as column vectors
function ShermanMorrison_Overlap(Oinv::Matrix{T}, phi1::AbstractVector{T}, phi2::AbstractVector{T}, b::T)::Tuple{T,Matrix{T}} where T
    # Use Sherman-Morrison formula to compute overlap ratio <phi1|B_V(x_i)|phi2> / <phi1|phi2>
    # O^-1 |phi1'>
    R = Oinv * phi1
    # <phi2| O^-1
    L = phi2' * Oinv
    # <phi2| O^-1 |phi1'>
    Gii = phi2' * R

    # O(x_i)/O = 1 + (b-1) * <phi2| O^-1 |phi1'>
    O_ratio = 1.0 + (b-1.0) * Gii

    # Update O^-1 by Sherman-Morrison formula
    factor = (1.0 - b) / (1.0 + (b-1.0) * Gii)
    Oinv_new = Oinv + factor * R * L

    return O_ratio, Oinv_new
end

function sampleOneSite(
phi1_up::AbstractVector{T}, 
phi1_dn::AbstractVector{T}, 
phi2_up::AbstractVector{T}, 
phi2_dn::AbstractVector{T}, 
Oinv_up::Matrix{T}, 
Oinv_dn::Matrix{T}, 
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64},
O::T,
x::Int
)::Tuple{Int,T,Matrix{T},Matrix{T}} where T
    @assert x == 1 || x == 2
    # Flip field
    x2 = 3-x
    @assert x2 == 1 || x2 == 2
    # Overlap for the new field
    O2_ratio_up, O2inv_up = ShermanMorrison_Overlap(Oinv_up, phi1_up, phi2_up, expV_up[x2]/expV_up[x])
    O2_ratio_dn, O2inv_dn = ShermanMorrison_Overlap(Oinv_dn, phi1_dn, phi2_dn, expV_dn[x2]/expV_dn[x])
    # Overall overlap with the new field
    O2_ratio = O2_ratio_up * O2_ratio_dn
    O2 = O2_ratio * O

    # Compute probability
    P = abs(O2) / (abs(O) + abs(O2))
    # Sampling
    rnd = rand()
    if rnd < P
        return x2, O2, O2inv_up, O2inv_dn
    else
        return x, O, Oinv_up, Oinv_dn
    end
end

# Sample x with probability <phi1|B_V(x)|phi2>
function sampleAuxField(
phi1_up::Matrix{T}, 
phi1_dn::Matrix{T}, 
phi2_up::Matrix{T}, 
phi2_dn::Matrix{T}, 
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64},
auxfld::Vector{Int},
toRight::Bool
)::Tuple{Matrix{T},Matrix{T},Float64,Vector{Int}} where T
    @assert size(phi1_up) == size(phi1_dn) == size(phi2_up) == size(phi2_dn)

    Nsites,Npar = size(phi1_up)

    # Apply the current fields to phi2
    phi2x_up = copy(phi2_up)
    phi2x_dn = copy(phi2_dn)
    applyV!(phi2x_up, auxfld, expV_up)
    applyV!(phi2x_dn, auxfld, expV_dn)

    # Compute overlap and inverse overlaps matrices
    Oup = phi1_up' * phi2x_up
    Odn = phi1_dn' * phi2x_dn
    Oinv_up = inv(Oup)
    Oinv_dn = inv(Odn)
    O = det(Oup) * det(Odn)

    # Sample
    xs = copy(auxfld)
    for i=1:Nsites
        # Sample xi
        #println("i = ",i)
        row1_up = @view phi1_up[i,:]
        row1_dn = @view phi1_dn[i,:]
        row2_up = @view phi2x_up[i,:]
        row2_dn = @view phi2x_dn[i,:]
        xs[i], O, Oinv_up, Oinv_dn = sampleOneSite(row1_up, row1_dn, row2_up, row2_dn, Oinv_up, Oinv_dn, expV_up, expV_dn, O, auxfld[i])
    end

    # Propagate phi
    if toRight
        phi_up = copy(phi1_up)
        phi_dn = copy(phi1_dn)
    else
        phi_up = copy(phi2_up)
        phi_dn = copy(phi2_dn)
    end
    applyV!(phi_up, xs, expV_up)
    applyV!(phi_dn, xs, expV_dn)

    # Stabilize the determinant
    phi_up = reOrthoDet(phi_up)
    phi_dn = reOrthoDet(phi_dn)

    return phi_up, phi_dn, O, xs
end

function sampleAuxField_old(
phi1_up::Matrix{T}, 
phi1_dn::Matrix{T}, 
phi2_up::Matrix{T}, 
phi2_dn::Matrix{T}, 
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64},
auxfld::Vector{Int},
toRight::Bool
)::Tuple{Matrix{T},Matrix{T},Float64,Vector{Int}} where T
    @assert size(phi1_up) == size(phi1_dn) == size(phi2_up) == size(phi2_dn)

    Nsites,Npar = size(phi1_up)

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

    # ----  Apply B_V ----
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
        # Check determinant overlap
        #@assert abs(O - overlap(phi_up, phi_dn, phi2_up, phi2_dn)) < 1e-12
    else
        phi_up = copy(phi2_up)
        phi_dn = copy(phi2_dn)
        applyV!(phi_up, auxfld, expV_up)
        applyV!(phi_dn, auxfld, expV_dn)

        # Check determinant overlap
        #@assert abs(O - overlap(phi1_up, phi1_dn, phi_up, phi_dn)) < 1e-12
    end

    # Stabilize the determinant
    phi_up = reOrthoDet(phi_up)
    phi_dn = reOrthoDet(phi_dn)

    return phi_up, phi_dn, O, auxfld
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

            # Measure at the center, between phi, phis[i+1]
            if (i == div(N,2))
                measure!(phi_up, phi_dn, phis_up[i+1], phis_dn[i+1], sign(O), obs, para)
            end


            phis_up[i] = phi_up
            phis_dn[i] = phi_dn
        end
    # Right to left
    else
        for i=N:-1:1
            phi_up, phi_dn, auxflds[i], O = sampleAuxField(phis_up[i-1], phis_dn[i-1], phis_up[i+1], phis_dn[i+1], auxflds[i],
                                                           expHk_half, expV_up, expV_dn; toRight=false)

            # Measure at the center, between phis[i], phi
            if (i-1 == div(N,2))
                measure!(phis_up[i-1], phis_dn[i-1], phi_up, phi_dn, sign(O), obs, para)
            end

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
    @assert abs(O/O_new - O_orth/O_orth_new) < 1e-10
    # Check sign
    @assert sign(O) == sign(O_orth)
    @assert sign(O_new) == sign(O_orth_new)
    # Check probability
    P = abs(O_new) / (abs(O) + abs(O_new))
    P_orth = abs(O_orth_new) / (abs(O_orth) + abs(O_orth_new))
    @assert abs(P - P_orth) < 1e-12
end
