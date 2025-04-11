import LinearAlgebra as linalg
include("Timer.jl")
using .Timer

function overlap(phi1::Matrix{T}, phi2::Matrix{T})::T where T
    return linalg.det(phi1' * phi2)
end

function detNorm(phi::Matrix{T})::Float64 where T
    O = overlap(phi,phi)
    return sqrt(O)
end

function normalizeDet!(phi::Matrix{T}) where T
    Npar = size(phi,2)
    for i=1:Npar
        norm = linalg.norm(phi[:,i])
        phi[:,i] ./= norm
        println(i," ",linalg.norm(phi[:,i]))
    end
end

function randomDet(Nsites::Int, Npar::Int; dtype::Type=Float64)
    A = rand(dtype, Nsites, Npar)
    Q = linalg.qr(A).Q
    return Q[:,1:Npar]
end

# Re-orthogonalize a determinant phi by a QR decomposition, phi=QR.
# Replace phi by Q.
# The reweighting factor det(R) is discarded.
# Note: det((Q'R')^\dagger QR) = det(Q'^\dagger Q) det(R')^* det(R)
function reOrthoDet!(phi::Matrix{T}) where T
    t = time_ns()

    F = linalg.qr(phi)
    phi .= Matrix(F.Q)

    timer["reOrtho"] += (time_ns() - t) / 1e9
end

# conf = {0, 0, 1, 0, 1, ...} is the occupations
function prodDet(conf::Vector{Int})::Matrix{Float64}
    # Check all the elements in conf are either 0 or 1
    @assert all(x -> x == 0 || x == 1, conf)

    Nsite = length(conf)
    Npar = sum(conf)
    phi = zeros(Nsite, Npar)

    ipar = 1
    for i=1:Nsite
        if conf[i] == 1
            phi[i,ipar] = 1.0
            ipar += 1
        end
    end
    return phi
end

# conf is a list of integer of 1, 2, 3, or 4
# 1: Emp, 2: Up, 3: Dn, 4: UpDn
# return up_conf and dn_conf = [0, 1, 0, ...] represent the occupations
function getConfUpDn(conf)
    N = length(conf)
    up_conf, dn_conf = zeros(Int,N), zeros(Int,N)
    for i=1:N
        if conf[i] == 2 || conf[i] == 4
            up_conf[i] = 1
        end
        if conf[i] == 3 || conf[i] == 4
            dn_conf[i] = 1
        end
    end
    return up_conf, dn_conf
end

function prodDetUpDn(conf::Vector{Int64})
    up_conf, dn_conf = getConfUpDn(conf)
    phi_up = prodDet(up_conf)
    phi_dn = prodDet(dn_conf)
    return phi_up, phi_dn
end
