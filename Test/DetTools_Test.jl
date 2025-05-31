include("../DetTools.jl")
include("../Measure.jl")
using Test

function reOrthoDet_GreensFunction_TEST(phi1, phi2)
    p1 = copy(phi1)
    p2 = copy(phi2)
    reOrthoDet!(p1)
    reOrthoDet!(p2)
    G = Greens_function(phi1, phi2)
    G_ortho = Greens_function(p1, p2)
    @assert norm(p1-phi1) > 1e-4
    @assert norm(p2-phi2) > 1e-4
    @test nomr(G-G_ortho) < 1e-12
end
