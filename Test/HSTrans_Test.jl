include("../H_k.jl")
include("../DetTools.jl")
include("../Measure.jl")
include("../EDTwoParticles.jl")
include("../HSTrans.jl")
using LinearAlgebra
using Test

function HS_expV_Test()
    dtau, U = 1.1, 2.0
    expV_up, expV_dn = HS_expV(dtau, U)
    ans_up = [5.837011501453833, 0.17132054643903455]
    ans_dn = [0.17132054643903455, 5.837011501453833]
    @test norm(expV_up-ans_up) < 1e-14
    @test norm(expV_dn-ans_dn) < 1e-14
end

function HS_applyV_Test()
    expV = [7., 8.]
    phi = [1. 2.;
           3. 4.;
           5. 6.]
    auxfld = [1, 2, 2]
    applyV!(phi, auxfld, expV)
    ans = [7.0 14.0; 24.0 32.0; 40.0 48.0]
    @test norm(phi - ans) < 1e-14
end

function HS_Test()
    Lx=4
    Ly=1
    tx=ty=1.0
    xpbc=false
    ypbc=false
    U = 2.0
    Nsites = Lx*Ly
    Npar = 2
    dtau = 0.1

    # Initialized phi
    phi_up0 = randomDet(Nsites, 1)
    phi_dn0 = randomDet(Nsites, 1)

    # ED
    psi = two_par_state(phi_up0, phi_dn0)
    HVED = U * H_nn(Nsites)
    HkED = -tx * H_hop(Nsites, xpbc)
    expHV = exp(-dtau*HVED)
    psiED = expHV * psi

    # Hubbard-Stratonovich transformation
    expV_up, expV_dn = HS_expV(dtau, U)
    d = Nsites^2
    psi2 = zeros(d)
    phis = []
    for i=1:2
      for j=1:2
        for k=1:2
          for l=1:2
            phi_up = copy(phi_up0)
            phi_dn = copy(phi_dn0)
            auxfld = [i,j,k,l]
            applyV!(phi_up, auxfld, expV_up)
            applyV!(phi_dn, auxfld, expV_dn)

            psi_i = two_par_state(phi_up, phi_dn)
            psi2 += psi_i

            push!(phis, [phi_up, phi_dn])
          end
        end
      end
    end
    psi2 *= exp(-0.5*dtau*U*Npar) * 0.5^Nsites

    diff = psiED - psi2
    @test norm(diff) < 1e-14


    # Test energy
    Hk = H_K(Lx,Ly,1,tx,ty,0.0,xpbc,ypbc,false)
    expHk_half = exp(-0.5*dtau*Hk)

    Ek, EV, O = 0.0, 0.0, 0.0
    for phi1 in phis
        for phi2 in phis
            phi1_up, phi1_dn = phi1
            phi2_up, phi2_dn = phi2

            # Measure
            G_up = Greens_function(phi1_up, phi2_up)
            G_dn = Greens_function(phi1_dn, phi2_dn)

            Oup = overlap(phi1_up, phi2_up)
            Odn = overlap(phi1_dn, phi2_dn)
            Oi = Oup*Odn

            Eki = kinetic_energy(G_up, G_dn, Hk) * Oi
            EVi = potential_energy(G_up, G_dn, U) * Oi


            Ek += Eki
            EV += EVi
            O += Oi
        end
    end
    
    Ek = Ek/O
    EV = EV/O

    O_ED = psiED' * psiED
    EV_ED = (psiED' * HVED * psiED)/O_ED
    Ek_ED = (psiED' * HkED * psiED)/O_ED

    @test abs(Ek-Ek_ED) < 1e-14
    @test abs(EV-EV_ED) < 1e-14
end

HS_expV_Test()
HS_applyV_Test()
HS_Test()
