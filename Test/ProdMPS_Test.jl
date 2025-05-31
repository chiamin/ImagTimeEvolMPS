include("../ProdMPS.jl")
include("../BinaryNumMPS.jl")
using ITensors
using Test

function length_Test()
    N = 10
    s = siteinds(2,N)
    chi = 4
    psi = random_mps(s;linkdims=chi)

    pmps = makeProdMPS(psi)
    @test length(pmps.conf) == length(pmps.As)
end

function binaryNum_MPS_ITensor(N::Int)
    bmps = binaryNum_MPS(N)

    s = siteinds("S=1/2",N)
    mps = MPS(s;linkdims=2)

    for i=1:N
        mps[i] = ITensor(bmps[i], inds(mps[i])...)
    end

    return mps
end

function ProdMPS_Test(N::Int)
    mps = binaryNum_MPS_ITensor(N)

    # Test getOverlap
    for i=1:10
        conf = rand(1:2, N)
        bval = getBinaryNum(conf)

        pmps = makeProdMPS(mps)
        val = getOverlap!(pmps, conf)

        @test abs(bval-val) < 1e-14
    end
end

length_Test()
ProdMPS_Test(4)
