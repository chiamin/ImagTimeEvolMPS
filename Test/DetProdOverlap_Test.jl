include("../DetProdOverlap.jl")
include("../DetTools.jl")
using Test

function DetProd_Spinless_Test()
    Nsites, Npar = 8, 4
    phi = randomDet(Nsites, Npar)

    occs = [3,5,7,8]
    # Compute the overlap in the standard way
    prod = prodDet(Nsites, Npar, occs)
    O = overlap(prod, phi)

    # Compute the overlap using detProdOverlap
    O2 = detProdOverlap(phi, occs)
    @test abs(O-O2) < 1e-14
end


function DetProd_Test()
    conf = [1,3,2,1,4,3,2,4]
    occ_up, occ_dn = get_occupied_sites(conf)

    Nsites = size(conf,1)
    Npar_up = size(occ_up,1)
    Npar_dn = size(occ_dn,1)
    phi_up = randomDet(Nsites, Npar_up)
    phi_dn = randomDet(Nsites, Npar_dn)

    # Compute the overlap in the standard way
    prod_up = prodDet(Nsites, Npar_up, occ_up)
    prod_dn = prodDet(Nsites, Npar_dn, occ_dn)
    O = overlap(phi_up, prod_up) * overlap(phi_dn, prod_dn)

    # Compute the overlap using detProdOverlap
    O2 = detProdOverlap(phi_up, phi_dn, conf)
    @test abs(O-O2) < 1e-14
end

@test get_occupied_sites([1,3,2,1,4,3,2,4]) == ([3,5,7,8], [2,5,6,8])
DetProd_Spinless_Test()
DetProd_Test()
