include("../DetTools.jl")

function prodDetUpDnRot(conf::Vector{Int64}, U_up::Matrix{T}, U_dn::Matrix{T}) where T
    up_sites, dn_sites = getOccupiedSites(conf)
    phi_up = U_up[:,up_sites]
    phi_dn = U_up[:,dn_sites]
    return phi_up, phi_dn
end
