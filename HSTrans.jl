
function HS_expV(dtau::Float64, U::Float64)::Tuple{Vector{Float64},Vector{Float64}}
    gamma = acosh(exp(0.5*dtau*U))
    expV_up = [exp(gamma),exp(-gamma)]
    expV_dn = [exp(-gamma),exp(gamma)]
    return expV_up, expV_dn
end

# Propagate a Slater determinant by the potential energy propagator exp(-deltau*V) with auxiliary fields
# phi: a spinless Slater determinant, stored in a matrix with shape (Nsites,Npar)
# aux_fids: auxiliary field, an array of dimension Nsites
# expV: 2-elements array, expV[x] = exp(gamma*s(sigma)*x_i) where x_i is the auxiliary field
function applyV!(phi::Matrix{T}, aux_flds::Vector{Int64}, expV::Vector{Float64}) where T
    Nsites,Npar = size(phi)

    @assert size(aux_flds) == (Nsites,)
    @assert size(expV) == (2,)

    for i=1:Nsites
        x = aux_flds[i]
        phi[i,:] *= expV[x]
    end
end

function applyExpH(
phi::Matrix{T}, 
auxfld::Vector{Int}, 
expHk_half::Matrix{Float64}, 
expV::Vector{Float64}
) where T
    Bphi = expHk_half * phi
    applyV!(Bphi, auxfld, expV)
    Bphi = expHk_half * Bphi
    return Bphi
end

function applyExpHExact(
    phi::Matrix{T},
    auxfld::Vector{Int},
    expHk_half::Matrix{Float64},
    dtau::Float64,
    U::Float64
) where T
    Nsites, Npar = size(phi)
    @assert length(auxfld) == Nsites
    @assert all(x -> x == 1 || x == 2, auxfld)

    λ = acosh(exp(0.5 * dtau * U))

    # Construct diagonal exp(V_HS(x)) with normalization and shift
    diagV = [exp((x == 1 ? λ : -λ) - 0.5 * dtau * U) for x in auxfld]

    # Apply diagonal potential to each row
    Vphi = copy(phi)
    for i in 1:Nsites
        Vphi[i, :] *= diagV[i]
    end

    # Sandwich with kinetic half-step propagators
    Bphi = expHk_half * Vphi
    Bphi = expHk_half * Bphi

    # Include exact normalization
    Bphi .*= 1.0 / (2.0 ^ Nsites)

    return Bphi
end
