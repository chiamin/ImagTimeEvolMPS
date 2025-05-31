include("../Hamiltonian.jl")
include("../SampleOcc.jl")
using Test

function RandomConf_Test()
    Nsites = 10
    Nup, Ndn = 3, 4
    conf = RandomConf(Nsites, Nup=Nup, Ndn=Ndn)
    Nup_t, Ndn_t = getParNums(conf)
    @test Nup == Nup_t
    @test Ndn == Ndn_t
end

RandomConf_Test()
@test_throws AssertionError RandomConf(1, Nup=2, Ndn=1)
