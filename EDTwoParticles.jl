include("H_k.jl")
include("DetTools.jl")
include("Hamiltonian.jl")

function get_basis(iup, idn, Nsites)
    return idn + Nsites*(iup-1)
end

function H_nn(Nsites)
    d = Nsites^2
    H = zeros(d,d)
    for i=1:Nsites
        ii = get_basis(i, i, Nsites)
        H[ii,ii] = 1.0
    end
    return H
end

function two_par_state(phi_up, phi_dn)
    Nsites = size(phi_up,1)
    d = Nsites^2
    phi = zeros(d)
    for i1=1:Nsites
        for i2=1:Nsites
            ii = get_basis(i1, i2, Nsites)
            phi[ii] = phi_up[i1] * phi_dn[i2]
        end
    end
    return phi
end

function H_hop(Nsites, pbc)
    d = Nsites^2
    H = zeros(d,d)
    for i_up=1:Nsites
        for i_dn=1:Nsites
            ii = get_basis(i_up, i_dn, Nsites)

            i2 = i_up+1
            if i2 > Nsites && pbc
                i2 -= Nsites
            end

            if i2 <= Nsites
                jj = get_basis(i2, i_dn, Nsites)
                H[ii,jj] = 1
                H[jj,ii] = 1
            end

            i2 = i_dn+1
            if i2 > Nsites && pbc
                i2 -= Nsites
            end

            if i2 <= Nsites
                jj = get_basis(i_up, i2, Nsites)
                H[ii,jj] = 1
                H[jj,ii] = 1
            end
        end
    end
    return H
end

function bin_to_int(auxfld)
    bin = auxfld.-1
    bstr = join(string.(bin))
    int = parse(Int, bstr; base=2)
    return int+1
end

function prob_dist(
phi1_up::AbstractMatrix, 
phi1_dn::AbstractMatrix, 
phi2_up::AbstractMatrix, 
phi2_dn::AbstractMatrix, 
expHk_half::Matrix{Float64},
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64},
)
    Nsites,Npar = size(phi1_up)
    phi1_up0 = expHk_half * phi1_up
    phi1_dn0 = expHk_half * phi1_dn
    phi2_up0 = expHk_half * phi2_up
    phi2_dn0 = expHk_half * phi2_dn

    xs, Os = [],[]
    for i=1:2
      for j=1:2
        for k=1:2
          for l=1:2
            for i2=1:2
              for j2=1:2
                for k2=1:2
                  for l2=1:2
                    phi1_up = phi1_up0
                    phi1_dn = phi1_dn0
                    phi2_up = phi2_up0
                    phi2_dn = phi2_dn0
                    auxfld1 = [i,j,k,l]
                    auxfld2 = [i2,j2,k2,l2]
                    x = bin_to_int(vcat(auxfld1, auxfld2))
                    phi1_up = applyV(phi1_up, auxfld1, expV_up)
                    phi1_dn = applyV(phi1_dn, auxfld1, expV_dn)
                    phi2_up = applyV(phi2_up, auxfld2, expV_up)
                    phi2_dn = applyV(phi2_dn, auxfld2, expV_dn)
                    phi1_up = expHk_half * phi1_up
                    phi1_dn = expHk_half * phi1_dn
                    phi2_up = expHk_half * phi2_up
                    phi2_dn = expHk_half * phi2_dn
                    O1 = overlap(phi1_up, phi2_up)
                    O2 = overlap(phi1_dn, phi2_dn)
                    O = O1*O2
                    push!(xs,x)
                    push!(Os,O)
                  end
                end
              end
            end
          end
        end
      end
    end
    return xs, Os
end

function profile(
phi1_up::AbstractMatrix, 
phi1_dn::AbstractMatrix, 
phi2_up::AbstractMatrix, 
phi2_dn::AbstractMatrix, 
expHk_half::Matrix{Float64},
expV_up::Vector{Float64}, 
expV_dn::Vector{Float64}; 
)
    auxfld = [1,1,1,1]
    data = zeros(16)
    for i=1:10000
        phi_up, phi_dn, auxfld, O = sampleAuxField(phi1_up, phi1_dn, phi2_up, phi2_dn, auxfld, expHk_half, expV_up, expV_dn; toRight=true)
        x = bin_to_int(auxfld)
        data[x+1] += 1
    end
    f = twinx()
    ff = plot(0:15, data, c="red")
    #display(f)
    return ff
end

# ----------------

function ED_GS(L, pbc, t, U)
    Hk = -t * H_hop(L, pbc)
    HV = U * H_nn(L)
    H = Hk + HV

    F = eigen(H)
    E0 = F.values[1]
    psi = F.vectors[:,1]
    return psi, E0, Hk, HV, H
end

# Get ground state as an MPS
function MPS_GS(L, pbc, t, U)
    sites = siteinds("Electron", L, conserve_qns=true)
    ampo = Hubbard(L, 1, t, t, U, 0, 0, 0, 0, pbc, false)
    H = MPO(ampo,sites)

    nsweeps = 100 # number of sweeps is 5
    maxdim = [20] # gradually increase states kept
    cutoff = [1e-14] # desired truncation error

    states = RandomState(L; Nup=1, Ndn=1)
    psi0 = MPS(sites, states)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    # Check energy
    psiED, E0, Hk, HV, HED = ED_GS(L, pbc, t, U)
    @test abs(E0-energy) < 1e-12

    # Fix the sign by the coefficient of |1_up 1_dn>
    # 1. determinant
    phi_up = zeros(L)
    phi_dn = zeros(L)
    phi_up[1] = 1.
    phi_dn[1] = 1.
    phi = two_par_state(phi_up, phi_dn)
    c1 = phi' * psiED
    # 2. MPS
    sites = siteinds(psi)
    state = fill("Emp",L)
    state[1] = "UpDn"
    prod = MPS(sites, state)
    c2 = inner(psi, prod)
    # Check the magnitudes are the same
    @test abs(abs(c1)-abs(c2)) < 1e-14
    # Fix the sign
    if (c1 > 0) != (c2 > 0)
        psi *= -1.
    end
    return psi
end

#<psi|Hk|phi> and <psi|HV|phi>
function ED_measure(L, pbc, t, U, phi_up, phi_dn)
    psi, E0, Hk, HV, H = ED_GS(L, pbc, t, U)
    phi = two_par_state(phi_up, phi_dn)

    Ek = psi' * Hk * phi
    EV = psi' * HV * phi
    O = psi' * phi
    @assert abs((Ek+EV)/O-E0) < 1e-14

    return Ek/O, EV/O, O
end

