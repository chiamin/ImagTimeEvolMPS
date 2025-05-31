using Test
include("../BinaryNumMPS.jl")

function binaryNum_MPS_Test(N::Int=4)
    mps = binaryNum_MPS(N)
    conf = rand(1:2, N)
    val = getBinaryNum(conf)
    mpsVal = getValue(mps, conf)
    @test abs(val-mpsVal) < 1e-14
end

binaryNum_MPS_Test(4)
