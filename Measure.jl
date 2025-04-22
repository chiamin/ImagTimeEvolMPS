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
end

function getObs(obs::Dict{String,Any}, name::String)
    return obs[name] / obs["count"]
end

function cleanObs!(obs::Dict{String,Any})
    empty!(obs)
    obs["count"] = 0.
end

function measure!(
    phi1_up::Matrix{T}, phi1_dn::Matrix{T},
    phi2_up::Matrix{T}, phi2_dn::Matrix{T},
    sign::Float64, obs::Dict{String,Any}, para::Dict{String,Any}
) where T
    Hk = para["Hk"]
    U = para["U"]

    G_up = Greens_function(phi1_up, phi2_up)
    G_dn = Greens_function(phi1_dn, phi2_dn)

    Ek = kinetic_energy(G_up, G_dn, Hk)
    EV = potential_energy(G_up, G_dn, U)
    nup, ndn = get_density(G_up, G_dn)

    measure!(obs, "Ek", Ek*sign)
    measure!(obs, "EV", EV*sign)
    measure!(obs, "nup", nup*sign)
    measure!(obs, "ndn", ndn*sign)
    measure!(obs, "sign", sign)

    # If "count" does not exist, set to 1; otherwise add 1
    if !haskey(obs,"count")
        obs["count"] = 0.
    end
    obs["count"] += 1.
end


# ======================== Debug functions ==============================
function test_GreensFunction(phi1, phi2)
    p1 = reOrthoDet(phi1)
    p2 = reOrthoDet(phi2)
    G = Greens_function(phi1, phi2)
    G_ortho = Greens_function(p1, p2)
    @assert norm(G-G_ortho) < 1e-12
end
