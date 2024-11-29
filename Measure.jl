using DataStructures

# Compute C_ij = <phi1| c_i^dagger c_j |phi2> / <phi1|phi2>
function Greens_function(phi1::Matrix{T}, phi2::Matrix{T})::Matrix{T} where T
    @assert size(phi1) == size(phi2)
    overlap_mat = phi1' * phi2
    inv_overlap = inv(overlap_mat)
    gf = phi2 * inv_overlap * phi1'
    return transpose(gf)
end

# Compute <phi1| H_k |phi2> / <phi1|phi2>
function kinetic_energy(G_up::Matrix{T}, G_dn::Matrix{T}, Hk::Matrix{Float64}) where T
    @assert size(G_up) == size(G_dn) == size(Hk)
    Ek = sum(Hk .* (G_up + G_dn))
    return Ek
end

function get_density(G_up::Matrix{T}, G_dn::Matrix{T}) where T
    nup = diag(G_up)
    ndn = diag(G_dn)
    return nup, ndn
end

# Compute <phi1| n_up n_dn |phi2> / <phi1|phi2>
function potential_energy(G_up::Matrix{T}, G_dn::Matrix{T}, U::Float64) where T
    nup, ndn = get_density(G_up, G_dn)
    nn = nup' * ndn
    return U*nn;
end

function measure!(obs::Dict{String,Any}, name::String, value)
    # Measure
    if !haskey(obs,name)
        obs[name] = value
    else
        obs[name] += value
    end

    # Store the name
    if !haskey(obs,"obss")
        obs["obss"] = Set{String}([name])
    else
        push!(obs["obss"], name)
    end
end

function getObs(obs::Dict{String,Any}, name::String)
    return obs[name] / obs["count"]
end

function cleanObs(obs::Dict{String,Any})
    for name in obs["obss"]
        delete!(obs, name)
    end
    obs["count"] = 0.
end

function measure!(phi1_up::Matrix{T}, phi1_dn::Matrix{T}, phi2_up::Matrix{T}, phi2_dn::Matrix{T}, w::Float64, obs::Dict{String,Any}) where T
    Hk = obs["Hk"]
    U = obs["U"]

    G_up = Greens_function(phi1_up, phi2_up)
    G_dn = Greens_function(phi1_dn, phi2_dn)

    Ek = kinetic_energy(G_up, G_dn, Hk)
    EV = potential_energy(G_up, G_dn, U)
    nup, ndn = get_density(G_up, G_dn)
    sign_w = sign(w)

    measure!(obs, "Ek", Ek*sign_w)
    measure!(obs, "EV", EV*sign_w)
    measure!(obs, "nup", nup*sign_w)
    measure!(obs, "ndn", ndn*sign_w)
    measure!(obs, "sign", sign_w)
    obs["count"] = get!(obs, "count", 0.) + 1.    # If "count" does not exist, set to 1; otherwise add 1
end
