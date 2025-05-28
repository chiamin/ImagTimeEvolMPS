using ITensors, ITensorMPS
using Optim
using LinearAlgebra, Test, Random

function RotOps(theta::Float64, spin="")
    @assert spin == "" || spin == "up" || spin == "dn"

    os = OpSum()
    # Single-particle rotation for each spin:
    # U12 = cos(theta) (c1^dag c1 + c2^dag c2 - 2*n1n2) + sin(theta) (c2^dag c1 - c1^dag c2)
    cos_up = cos(theta)
    sin_up = sin(theta)
    os += cos_up, "N"*spin, 1
    os += cos_up, "N"*spin, 2
    os += sin_up, "Cdag"*spin, 2, "C"*spin, 1
    os += -sin_up, "Cdag"*spin, 1, "C"*spin, 2
    os += -2*cos_up,"N"*spin,1,"N"*spin,2

    # P00 = (1-n1)(1-n2) = c1 c1^dag c2 c2^dag
    os += 1., "C"*spin,1,"Cdag"*spin,1,"C"*spin,2,"Cdag"*spin,2
    # P11 = n1n2 = c1^dag c1 c2^dag c2
    os += 1., "Cdag"*spin,1,"C"*spin,1,"Cdag"*spin,2,"C"*spin,2

    return os
end

function RotMat(theta)
    M = zeros(2,2)
    M[1,1] = M[2,2] = cos(theta)
    M[1,2] = -sin(theta)
    M[2,1] = sin(theta)
    return M
end

function RotGate(sites, theta, spin="")
    @assert length(sites) == 2

    os = RotOps(theta, spin)
    UMPO = MPO(os, sites)
    U = UMPO[1] * UMPO[2]
    return U
end

function RotGate(s1::Index, s2::Index, theta_up::Float64, theta_dn::Float64)
    s = [s1, s2]
    Gup = RotGate(s, theta_up, "up")
    Gdn = RotGate(s, theta_dn, "dn")

    Gdn = prime(Gdn)
    G = Gup * Gdn
    ITensors.replaceprime!(G, 2, 1)
    return G
end

function applyRot(psi::MPS, i1::Int, i2::Int, thetas::Vector{Float64})
    @assert orthocenter(psi) == i1 || orthocenter(psi) == i2
    s1 = siteind(psi, i1)
    s2 = siteind(psi, i2)
    G = RotGate(s1, s2, thetas[1], thetas[2])
    wf = (psi[i1] * psi[i2]) * G
    wf = noprime(wf)
    return wf
end

function applyRot!(psi::MPS, i1::Int, i2::Int, thetas::Vector{Float64}; cutoff=1e-14)
    wf = applyRot(psi, i1, i2, thetas)

    inds3 = uniqueinds(psi[i1], psi[i2])
    U, S, V = svd(wf, inds3; cutoff=cutoff)
    psi[i1] = U
    psi[i2] = S * V
end

function costFunc(psi::MPS, i1::Int, i2::Int, thetas::Vector{Float64}, maxdim::Int; cutoff=1e-14)
    wf = applyRot(psi, i1, i2, thetas)
    inds3 = uniqueinds(psi[i1],psi[i2])
    U,S,V = svd(wf,inds3,cutoff=cutoff)

    err = 0.0
    for n=maxdim+1:dim(S, 1)
      #p = S[n,n]^2
      err += S[n,n]
    end
    return err
end

function getBondDims(psi)
    bonddims = Vector{Int}()
    for i=1:length(psi)-1
        ii = linkind(psi, i)
        push!(bonddims, dim(ii))
    end
    return bonddims
end

function OptimizeBasis(psi::MPS; nIter=2, cutoff=1e-8, targetDim=2)
    psi = copy(psi)

    # Optimize basis
    i1 = 0
    i2 = 0
    targetdim = 0

    # Initial bond dimensions
    dims = getBondDims(psi)
    println("Initial bond dimensions: ",dims)

    # Set up the boundaries and the initial values of the rotation angles
    lower = [0.0, 0.0]
    upper = [2.0*pi, 2.0*pi]
    initial = [0.1*pi, 0.1*pi]

    # Set up the optimizor's options
    options = Optim.Options(
        iterations = 40,
        f_reltol = 1e-8,
        g_abstol = 1e-8,
        x_abstol = 1e-8,
        outer_x_reltol = 1e-8,
        outer_f_reltol = 1e-8,
        outer_g_abstol = 1e-8,
        outer_iterations = 40,
        show_trace = false
    )

    # Do the optimization
    N = length(psi)
    # The overall unitary matrix. The column indices are for the physical sites.
    U_up = Matrix{Float64}(I, N, N)
    U_dn = Matrix{Float64}(I, N, N)
    for n=1:nIter
        for i in [1:N-1; N-2:-1:2]
            global i1 = i
            global i2 = i+1
            orthogonalize!(psi, i1)

            # Define the cost function
            function myf(thetas::Vector{Float64})::Float64
                return costFunc(psi, i1, i2, thetas, targetDim)
            end

            # Optimize theta
            result = optimize(myf, lower, upper, initial, Fminbox(LBFGS()), options)
            min_x = Optim.minimizer(result)

            # Apply rotation
            applyRot!(psi, i1, i2, min_x; cutoff=cutoff)
            println(i," ",targetDim," ",min_x,": ",getBondDims(psi))

            # Update the overall unitary matrix
            R_up = RotMat(min_x[1])
            R_dn = RotMat(min_x[2])
            U_up[i1:i2, :] .= R_up * U_up[i1:i2, :]
            U_dn[i1:i2, :] .= R_dn * U_dn[i1:i2, :]
        end
    end

    # Columns of U_up and U_dn are the new orbitals
    U_up = U_up'
    U_dn = U_dn'

    return psi, U_up, U_dn
end

