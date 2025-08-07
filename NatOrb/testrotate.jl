#!/Users/srw/.juliaup/bin/julia
include("GivensVijkl.jl")
using ITensors, ITensorMPS, LinearAlgebra
#using GivensVijkl
using HDF5

function eig(M::Matrix{Float64})
    F = eigen(Symmetric(M))
    F.values,F.vectors
end
function diagrho(rho)
    eval,evec = eig(-rho)
    for rr = 1:length(eval)
        sum(evec[:,rr]) < 0.0 && (evec[:,rr] *= -1.0)
    end
    -eval,evec
end

let
Nx = Ny = 4
N = Nx * Ny
t = 1.0
U  = 8.0
Nup = Ndn = 7
sites = siteinds("Electron",N; conserve_qns=true)
snakeind(ix,iy) = (ix-1)*Ny + iy

ampo = AutoMPO()
for i=1:N
    ampo += (U,"Nupdn",i)
end
for ix=1:Nx, iy=1:Ny
    i = snakeind(ix,iy)
    for jx=1:Nx, jy=1:Ny
        j = snakeind(jx,jy)
        if (ix-jx)^2+(iy-jy)^2 == 1
            ampo += (-t,"Cdagup",i,"Cup",j)
            ampo += (-t,"Cdagup",j,"Cup",i)
            ampo += (-t,"Cdagdn",i,"Cdn",j)
            ampo += (-t,"Cdagdn",j,"Cdn",i)
        end
    end
end
H = MPO(ampo,sites)

maxdim = [20,20,30,30,50,70,100,150,200,300,400,600,600,900,900,1200,1200]
nsweeps = length(maxdim)
cutoff = 1e-11
noise = [1e-7]

state = ["Emp" for n=1:N]
state[1:2:2*Nup-1] .= "Up"
state[2:2:2*Ndn] .= "Dn"
psi = MPS(sites,state)
energy, psi = dmrg(H,psi; nsweeps, cutoff, maxdim,noise)
@show energy

rhoup = correlation_matrix(psi,"Cdagup","Cup")
rhodn = correlation_matrix(psi,"Cdagdn","Cdn")

println("diff norm = ",norm(rhoup-rhodn))
eval_up,U_up = eig(rhoup)
eval_dn,U_dn = eig(rhodn)


rhotot = rhoup+rhodn
occs,nos = diagrho(rhotot)
for i=1:N
    println(i,"\t",occs[i])
end

gates,sign = getgates(nos;reportsign = true)
@show sign

npsi = applygates(gates,psi;gatesign=sign,maxdim=4000)
@show maxlinkdim(npsi)
@show norm(npsi)
normalize!(npsi)
ntot = expect(npsi,"Ntot")
for i=1:N
    println(i,"\t",ntot[i])
end
orthogonalize!(npsi,1)
for i=1:5
    v = sample(npsi)
    @show v
end

f = h5open("psi.h5","w")
write(f,"U_up",U_up)
write(f,"U_dn",U_dn)
write(f,"U",nos)
write(f,"psi",npsi)
close(f)

end
