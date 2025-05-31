include("../H_k.jl")
include("../DetTools.jl")
include("../Measure.jl")
include("../EDTwoParticles.jl")
using LinearAlgebra
using Test

function HkED_Test()
    Lx=4
    Ly=1
    tx=ty=1.0
    xpbc=false
    ypbc=false
    Nsites = Lx*Ly

    Hk = H_K(Lx,Ly,1,tx,ty,0.0,xpbc,ypbc,false)
    HkED = -tx * H_hop(Nsites, xpbc)

    for i=1:Nsites
        phi_up = zeros(Nsites)
        phi_up[i] = 1.0
        for j=1:Nsites
            phi_dn = zeros(Nsites)
            phi_dn[j] = 1.0

            phi2_up = Hk * phi_up
            phi2_dn = Hk * phi_dn
            psi2_up = two_par_state(phi2_up, phi_dn)
            psi2_dn = two_par_state(phi_up, phi2_dn)
            psi2 = psi2_up + psi2_dn

            psi = two_par_state(phi_up, phi_dn)
            psi2ed = HkED * psi

            dpsi = psi2 - psi2ed
            @test norm(dpsi) < 1e-14
        end
    end
end

HkED_Test()
