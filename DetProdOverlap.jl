using LinearAlgebra

function getOccupiedSites(conf::Vector{Int64})::Tuple{Vector{Int64},Vector{Int64}}
    up_sites = findall(x -> x == 2 || x == 4, conf)
    dn_sites = findall(x -> x == 3 || x == 4, conf)
    return up_sites, dn_sites
end

function occSitesToConf(occ_up::Vector{Int}, occ_dn::Vector{Int}, Nsites::Int)
    conf = ones(Nsites)
    for iup in occ_up
        conf[iup] = 2
    end
    for idn in occ_dn
        if conf[idn] == 1
            conf[idn] = 3
        else
            conf[idn] = 4
        end
    end
    return conf
end

function detProdOverlap(phi::Matrix{T}, occ_sites::Vector{Int64})::T where T
    mat = phi[occ_sites,:]
    return det(mat)
end

# conf: Configuration for electron Hilbert space.
#       conf = [c1, c2, c3, ...], where ci = 0, 1, 2 or 3 for Emp, Up, Dn, and UpDn
function detProdOverlap(phi_up::Matrix{T}, phi_dn::Matrix{T}, conf::Vector{Int64})::T where T
    up_sites, dn_sites = getOccupiedSites(conf)

    Nsites_up, Npar_up = size(phi_up)
    Nsites_dn, Npar_dn = size(phi_dn)
    @assert Nsites_up == Nsites_dn
    @assert size(up_sites) == (Npar_up,)
    @assert size(dn_sites) == (Npar_dn,)

    Oup = detProdOverlap(phi_up, up_sites)
    Odn = detProdOverlap(phi_dn, dn_sites)
    return Oup * Odn
end

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
