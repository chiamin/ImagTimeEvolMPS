include("../DetProdOverlap.jl")

function detProdRotOverlap(phi::Matrix{T}, occ_sites::Vector{Int64}, U::Matrix{T})::T where T
    phi2 = U[:,occ_sites]
    return overlap(phi,phi2)
end

function detProdRotOverlap(phi_up::Matrix{T}, phi_dn::Matrix{T}, conf::Vector{Int64}, U_up::Matrix{T}, U_dn::Matrix{T})::T where T
    up_sites, dn_sites = getOccupiedSites(conf)
    o_up = detProdRotOverlap(phi_up, up_sites, U_up)
    o_dn = detProdRotOverlap(phi_dn, dn_sites, U_dn)
    return o_up * o_dn
end
