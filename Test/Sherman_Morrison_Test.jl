import LinearAlgebra
using Test
include("../SampleDet.jl")
using Random
using BenchmarkTools

function SMTest()
    # Exact
    N, M = 10, 4
    phi1_up = rand(N,M)
    phi2_up = rand(N,M)
    phi1_dn = rand(N,M)
    phi2_dn = rand(N,M)
    O_mat_up = phi1_up' * phi2_up
    O_mat_dn = phi1_dn' * phi2_dn
    O_up = linalg.det(O_mat_up)
    O_dn = linalg.det(O_mat_dn)
    Oinv_up = inv(O_mat_up)
    Oinv_dn = inv(O_mat_dn)
    # Apply fields
    expV_up = rand(2)
    expV_dn = rand(2)

    site = 1
    phi1x1_up = copy(phi1_up)
    phi1x1_dn = copy(phi1_dn)
    phi1x2_up = copy(phi1_up)
    phi1x2_dn = copy(phi1_dn)
    phi1x1_up[site,:] *= expV_up[1]
    phi1x1_dn[site,:] *= expV_dn[1]
    Ox1_up = overlap(phi1x1_up, phi2_up)
    Ox1_dn = overlap(phi1x1_dn, phi2_dn)
    phi1x2_up[site,:] *= expV_up[2]
    phi1x2_dn[site,:] *= expV_dn[2]
    Ox2_up = overlap(phi1x2_up, phi2_up)
    Ox2_dn = overlap(phi1x2_dn, phi2_dn)
    println("O(x=1): ",Ox1_up/O_up," ",Ox1_dn/O_dn)
    println("O(x=2): ",Ox2_up/O_up," ",Ox2_dn/O_dn)

    O_mat_up_x1 = phi1x1_up' * phi2_up
    O_mat_up_x2 = phi1x2_up' * phi2_up
    O_mat_dn_x1 = phi1x1_dn' * phi2_dn
    O_mat_dn_x2 = phi1x2_dn' * phi2_dn
    Oinv_up_xx = [inv(O_mat_up_x1), inv(O_mat_up_x2)]
    Oinv_dn_xx = [inv(O_mat_dn_x1), inv(O_mat_dn_x2)]


    # Use Sherman-Morrison formula to compute overlap ratio <phi1|B_V(x_site)|phi2> / <phi1|phi2>
    # O^-1 |phi1'(site)>
    R_up = Oinv_up * phi1_up'[:,site]
    R_dn = Oinv_dn * phi1_dn'[:,site]
    # <phi2(site)| O^-1
    L_up = transpose(phi2_up[site,:]) * Oinv_up
    L_dn = transpose(phi2_dn[site,:]) * Oinv_dn
    # <phi2(site)| O^-1 |phi1'(site)>
    Gii_up = transpose(phi2_up[site,:]) * R_up
    Gii_dn = transpose(phi2_dn[site,:]) * R_dn
    # (expV(x)-1) * <phi2(site)| O^-1 |phi1'(site)>
    Gx_up = [(expV_up[1]-1.0) * Gii_up, (expV_up[2]-1.0) * Gii_up]
    Gx_dn = [(expV_dn[1]-1.0) * Gii_dn, (expV_dn[2]-1.0) * Gii_dn]
    # <phi1|B_V(x_site)|phi2> / <phi1|phi2> = 1 + Gx_up[x]
    O1_up = 1.0 + Gx_up[1]
    O2_up = 1.0 + Gx_up[2]
    O1_dn = 1.0 + Gx_dn[1]
    O2_dn = 1.0 + Gx_dn[2]
    println("O(x=1): ",O1_up," ",O1_dn)
    println("O(x=2): ",O2_up," ",O2_dn)

    @test abs(O1_up - Ox1_up/O_up) < 1e-12
    @test abs(O2_up - Ox2_up/O_up) < 1e-12
    @test abs(O1_dn - Ox1_dn/O_dn) < 1e-12
    @test abs(O2_dn - Ox2_dn/O_dn) < 1e-12

    # <phi1|B_V(x)|phi2> / <phi1|phi2> = 1 + Gx_up[x]
    O1 = O1_up * O1_dn
    O2 = O2_up * O2_dn

    # Sample field
    P1 = O1/(O1+O2)
    x = 2
    if rand() < P1
        x = 1
    end

    toRight = true
    for x=1:2
        # Update phi
        if toRight
            # Propagate phi1
            phi1_up[site,:] *= expV_up[x]
            phi1_dn[site,:] *= expV_dn[x]
        else
            # Propagate phi2
            phi2_up[site,:] *= expV_up[x]
            phi2_dn[site,:] *= expV_dn[x]
        end

        # Update O^-1 by Sherman-Morrison formula
        factor_up = (1.0 - expV_up[x]) / (1.0 + Gx_up[x])
        factor_dn = (1.0 - expV_dn[x]) / (1.0 + Gx_dn[x])
        Oinv_up_x = Oinv_up + factor_up * R_up * L_up
        Oinv_dn_x = Oinv_dn + factor_dn * R_dn * L_dn

        println(linalg.norm(Oinv_up_xx[x] - Oinv_up_x)," ",linalg.norm(Oinv_dn_xx[x] - Oinv_dn_x))
        @test norm(Oinv_up_xx[x] - Oinv_up_x) < 1e-8
        @test norm(Oinv_dn_xx[x] - Oinv_dn_x) < 1e-8
    end
end
using LinearAlgebra, Random, Test

function test_ShermanMorrison_Overlap()
    Random.seed!(1234)  # For reproducibility
    N = 6  # Number of rows (sites)
    M = 4  # Number of columns (orbitals)
    
    # Generate random Slater determinant matrices
    phi1 = randn(Float64, N, M)
    phi2 = randn(Float64, N, M)

    # Compute original overlap matrix and inverse
    O_mat = phi1' * phi2
    O = det(O_mat)
    Oinv = inv(O_mat)

    # Choose a site to apply B_V
    site = 2
    b = 1.5  # Some scaling factor

    # Clone phi2 and apply B_V only to the specified site
    phi2_b = copy(phi2)
    phi2_b[site, :] *= b

    # Compute new overlap and inverse exactly
    O_mat_b = phi1' * phi2_b
    O_b_exact = det(O_mat_b)
    Oinv_b_exact = inv(O_mat_b)

    # Now test with Sherman–Morrison
    phi1_row = phi1[site, :]  # Column vector
    phi2_row = phi2[site, :]  # Column vector

    O_ratio, Oinv_b = ShermanMorrison_Overlap(Oinv, phi1_row, phi2_row, b)

    # Reconstruct O_b from ratio
    O_b = O * O_ratio

    println("Exact det O_b      = ", O_b_exact)
    println("SM     det O_b     = ", O_b)
    println("Relative error     = ", abs(O_b - O_b_exact) / abs(O_b_exact))

    println("Inv O_b error norm = ", norm(Oinv_b - Oinv_b_exact))

    @test isapprox(O_b, O_b_exact; rtol=1e-12)
    @test isapprox(Oinv_b, Oinv_b_exact; rtol=1e-12)
end
using LinearAlgebra, Random, Test

function test_sampleOneSite()
    Random.seed!(42)
    N = 6  # number of sites
    M = 4  # number of orbitals

    # Generate real-valued Slater determinants
    phi1_up = randn(Float64, N, M)
    phi1_dn = randn(Float64, N, M)
    phi2_up = randn(Float64, N, M)
    phi2_dn = randn(Float64, N, M)

    # Set up two possible HS exponentials
    expV_up = rand(2) .+ 1.0
    expV_dn = rand(2) .+ 1.0

    # Apply 
    x = 2
    site = 3
    phi2_up_x1 = copy(phi2_up)
    phi2_dn_x1 = copy(phi2_dn)
    phi2_up_x1[site, :] *= expV_up[x]
    phi2_dn_x1[site, :] *= expV_dn[x]

    # Compute overlap matrices and determinants
    O_mat_up = phi1_up' * phi2_up_x1
    O_mat_dn = phi1_dn' * phi2_dn_x1
    O_up = det(O_mat_up)
    O_dn = det(O_mat_dn)
    O = O_up * O_dn
    Oinv_up = inv(O_mat_up)
    Oinv_dn = inv(O_mat_dn)

    # Select a site to apply the field
    phi1_row_up = phi1_up[site, :]
    phi1_row_dn = phi1_dn[site, :]
    phi2_row_up = phi2_up_x1[site, :]
    phi2_row_dn = phi2_dn_x1[site, :]

    # Sample
    x_new, O_new, Oinv_up_new, Oinv_dn_new = sampleOneSite(
        phi1_row_up, phi1_row_dn,
        phi2_row_up, phi2_row_dn,
        Oinv_up, Oinv_dn,
        expV_up, expV_dn,
        O, x
    )

    # Apply field directly to phi2
    phi2_up_x2 = copy(phi2_up)
    phi2_dn_x2 = copy(phi2_dn)
    phi2_up_x2[site, :] *= expV_up[x_new]
    phi2_dn_x2[site, :] *= expV_dn[x_new]

    # Exact recomputation
    O_mat_up_x = phi1_up' * phi2_up_x2
    O_mat_dn_x = phi1_dn' * phi2_dn_x2
    O_exact = det(O_mat_up_x) * det(O_mat_dn_x)
    Oinv_up_exact = inv(O_mat_up_x)
    Oinv_dn_exact = inv(O_mat_dn_x)

    println("x = $x → sampled x = $x_new")
    println("Overlap: computed = $O_new, exact = $O_exact")
    println("Inverse up error norm: ", norm(Oinv_up_new - Oinv_up_exact))
    println("Inverse dn error norm: ", norm(Oinv_dn_new - Oinv_dn_exact))

    @test isapprox(O_new, O_exact; rtol=1e-12)
    @test isapprox(Oinv_up_new, Oinv_up_exact; rtol=1e-12)
    @test isapprox(Oinv_dn_new, Oinv_dn_exact; rtol=1e-12)
end

function SampleALL_Test()
    N, M = 32, 14
    phi1_up = rand(N,M)
    phi2_up = phi1_up
    phi1_dn = rand(N,M)
    phi2_dn = phi1_dn
    expV_up = rand(2)
    expV_dn = rand(2)
    toRight = true
    auxfld = fill(1,N)

    phi1_up_bk = deepcopy(phi1_up)

    Random.seed!(123)
    @time phi_up, phi_dn, O, x = sampleAuxField(phi1_up, phi1_dn, phi2_up, phi2_dn, expV_up, expV_dn, auxfld, toRight)

    @assert norm(phi1_up_bk - phi1_up) < 1e-14

    Random.seed!(123)
    @time phi_up2, phi_dn2, O2, x2 = sampleAuxField_old(phi1_up, phi1_dn, phi2_up, phi2_dn, expV_up, expV_dn, auxfld, toRight)

    println("Final overlap: ",O," ",O2)
    println("x: ",x)
    println("x: ",x2)

    @test norm(phi_up - phi_up2) < 1e-12
    @test norm(phi_dn - phi_dn2) < 1e-12
    @test isapprox(O, O2, rtol=1e-12)
    @test sum(x .- x2) == 0
end

#test_ShermanMorrison_Overlap()
#test_sampleOneSite()
#SMTest()
SampleALL_Test()
