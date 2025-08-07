module GivensVijkl
using ITensors, ITensorMPS, LinearAlgebra
include("Util.jl")
using .Util
using PrecompileTools: @compile_workload    # this is a small dependency

export applygates, getgates, getVijkl, checkgates, getVijklupdn

# Gate = (angle, i, j)

function get1gate(vj,vj1,sign=1.0)
    ang = (vj1 == 0.0 ? pi*0.5 : atan(vj/vj1)) * sign
    c,s = cos(ang), sin(ang)
    [c s; -s c],ang
end

# destinations are the sites each orbital should now sit on
function getgates(orbitals,destinations::Vector{Int})
    n,m = size(orbitals)
    @show n,m
    orbs = copy(orbitals)
    U = eye(n)
    gates = []
    for i=1:m
        di = destinations[i]
        v = orbs[:,i]
        gat = []
        for j=n:-1:di+1
            mat,ang = get1gate(v[j],v[j-1])
            ra = j-1:j
            v[ra] .= mat * v[ra]
            U[ra,:] .= mat * U[ra,:]
            orbs[ra,:] .= mat * orbs[ra,:]
            push!(gat,(ang,j-1,j))
        end
        rgat = []
        for j=length(gat)-1:-1:1
            ang,k,kp = -gat[j][1],gat[j][2],gat[j][3]
            push!(rgat,(ang,k,kp))
            c,s = cos(ang), sin(ang)
            mat = [c s; -s c]
            ra = k:kp
            v[ra] .= mat * v[ra]
            U[ra,:] .= mat * U[ra,:]
            orbs[ra,:] .= mat * orbs[ra,:]
        end
	if v[di] < 0.0
	    j = i+1
            ra = j-1:j
	    ang = pi
            c,s = cos(ang), sin(ang)
            mat = [c s; -s c]
            v[ra] .= mat * v[ra]
            U[ra,:] .= mat * U[ra,:]
            orbs[ra,:] .= mat * orbs[ra,:]
            push!(gat,(ang,j-1,j))
	end
        gates = vcat(gates,gat,rgat)
        gat = []
        for j=1:di-1
            mat,ang = get1gate(v[j],v[j+1],-1.0)
            ra = j:j+1
            v[ra] .= mat * v[ra]
            U[ra,:] .= mat * U[ra,:]
            orbs[ra,:] .= mat * orbs[ra,:]
            push!(gat,(ang,j,j+1))
        end
        rgat = []
        for j=length(gat)-1:-1:1
            ang,k,kp = -gat[j][1],gat[j][2],gat[j][3]
            push!(rgat,(ang,k,kp))
            c,s = cos(ang), sin(ang)
            mat = [c s; -s c]
            ra = k:kp
            v[ra] .= mat * v[ra]
            U[ra,:] .= mat * U[ra,:]
            orbs[ra,:] .= mat * orbs[ra,:]
        end
	if v[di] < 0.0
	    j = i-1
            ra = j:j+1
	    ang = pi
            c,s = cos(ang), sin(ang)
            mat = [c s; -s c]
            v[ra] .= mat * v[ra]
            U[ra,:] .= mat * U[ra,:]
            orbs[ra,:] .= mat * orbs[ra,:]
            push!(gat,(ang,j,j+1))
	end
        gates = vcat(gates,gat,rgat)

    end
    chorb = U * orbitals
    f = open("orbs.dat","w")
    for i=1:m
        for j=1:n
            printspf(f,j,chorb[j,i])
        end
        printspf(f,"@")
    end
    close(f)

    gates
end

# put orbitals into one site each, starting at 1
function getgates(orbitals;reportsign = false)
    n,m = size(orbitals)
    orbs = copy(orbitals)
    U = eye(n)
    gates = []
    for i=1:m
        v = orbs[:,i]
        for j=n:-1:i+1
            mat,ang = get1gate(v[j],v[j-1])
            v[j-1:j] .= mat * v[j-1:j]
            U[j-1:j,:] .= mat * U[j-1:j,:]
            orbs[j-1:j,:] .= mat * orbs[j-1:j,:]
            push!(gates,(ang,j-1,j))
        end
	if v[i] < 0.0
	    j = i+1
            j > n && continue
	    ang = pi
            c,s = cos(ang), sin(ang)
            mat = [c s; -s c]
            v[j-1:j] .= mat * v[j-1:j]
            U[j-1:j,:] .= mat * U[j-1:j,:]
            orbs[j-1:j,:] .= mat * orbs[j-1:j,:]
            push!(gates,(ang,j-1,j))
	    #@show v[i]
	end
    end
    #chorb = U * orbitals
    #@show chorb
    dif = checkgates(gates,orbitals)
    if reportsign
        if dif > 1.0e-10
            if abs(dif-2.0) > 1.0e-8
                error("unexpected dif in getgates")
            end
            return gates,-1
        else
            return gates,1
        end
    end
    gates
end

function ITensors.op(::OpName"Givens", ::SiteType"Electron", s1::Index, s2::Index; θ)
  ampo = AutoMPO()
  ampo += (θ, "Cdagup", 1, "Cup", 2)
  ampo -= (θ, "Cdagup", 2, "Cup", 1)
  ampo += (θ, "Cdagdn", 1, "Cdn", 2)
  ampo -= (θ, "Cdagdn", 2, "Cdn", 1)
  return prod(MPO(ampo, [s1, s2]))
end

#g = exp(op("Givens", s1, s2; θ = π/8))


function applygates(gates,psi;doreverse = false,cutoff=1e-8, maxdim=2000,gatesign=1)
    s = siteinds(psi)
    ga = [exp(op("Givens",s[g[2]],s[g[3]]; θ = (doreverse ? -g[1] : g[1]))) for g in gates]
    doreverse && (ga = reverse(ga))
    npsi = apply(ga,psi; cutoff,maxdim)
    if gatesign == -1
        n = length(npsi)
        orthogonalize!(npsi,n)
        npsi[n] = noprime(op("F",s[n]) * npsi[n])
    end
    npsi
end

function checkgates(gates,orbitals)
    n,m = size(orbitals)
    U = eye(n)
    for g in gates
        ang,i,j = g
        c,s = cos(ang), sin(ang)
        mat = [c s; -s c]
        #U[[i,j],:] .= mat * U[[i,j],:]
        U[:,[i,j]] .= U[:,[i,j]] * mat'
    end
    #display(U)
    #display(orbitals)
    #@show norm(U[:,1:m]-orbitals)
    norm(U[:,1:m]-orbitals)
end

function getVijkl(newbas,V)
    N,Nijkl = size(newbas)
    prodinds = []
    prodrev = fill(0,Nijkl,Nijkl)
    for i=1:Nijkl, j=i:Nijkl
        push!(prodinds,(i,j))
        prodrev[i,j] = prodrev[j,i] = length(prodinds)
    end
    npr = length(prodinds)
    prod = zeros(N,npr)
    for ij=1:npr
        ij1,ij2 = prodinds[ij][1:2]
        prod[:,ij] = newbas[:,ij1] .* newbas[:,ij2]
    end
    #prodchk = [newbas[k,prodinds[ij][1]] * newbas[k,prodinds[ij][2]] for k=1:N, ij=1:npr]
    #@show norm(prodchk-prod),"remove me"
    Vpr = prod' * V * prod
    Vijkl = zeros(Nijkl,Nijkl,Nijkl,Nijkl)
    for k=1:Nijkl, l=1:Nijkl
        prkl = prodrev[k,l]
        for j=1:Nijkl, i=1:Nijkl
            Vijkl[i,j,k,l] = Vpr[prodrev[i,j],prkl]
        end
    end
    #Vijklchk = [Vpr[prodrev[i,j],prodrev[k,l]] for i=1:Nijkl, j=1:Nijkl, k=1:Nijkl, l=1:Nijkl]
    #@show norm(Vijklchk-Vijkl),"remove me"
    Vijklp = permutedims(Vijkl,[1,3,4,2])
    Vijkl,Vijklp        # cdag c cdag c;  cdag cdag c c
end
function getVijklupdn(basup,basdn,V)
    N,Nup = size(basup)
    N,Ndn = size(basdn)
    prodindsup = []
    prodrevup = fill(0,Nup,Nup)
    for i=1:Nup, j=i:Nup
        push!(prodindsup,(i,j))
        prodrevup[i,j] = prodrevup[j,i] = length(prodindsup)
    end
    nprup = length(prodindsup)
    produp = zeros(N,nprup)
    for ij=1:nprup
        ij1,ij2 = prodindsup[ij][1:2]
        produp[:,ij] = basup[:,ij1] .* basup[:,ij2]
    end
    prodindsdn = []
    prodrevdn = fill(0,Ndn,Ndn)
    for i=1:Ndn, j=i:Ndn
        push!(prodindsdn,(i,j))
        prodrevdn[i,j] = prodrevdn[j,i] = length(prodindsdn)
    end
    nprdn = length(prodindsdn)
    proddn = zeros(N,nprdn)
    for ij=1:nprdn
        ij1,ij2 = prodindsdn[ij][1:2]
        proddn[:,ij] = basdn[:,ij1] .* basdn[:,ij2]
    end

    Vpr = produp' * V * proddn
    Vijkl = zeros(Nup,Nup,Ndn,Ndn)
    for k=1:Ndn, l=1:Ndn
        prkl = prodrevdn[k,l]
        for j=1:Nup, i=1:Nup
            Vijkl[i,j,k,l] = Vpr[prodrevup[i,j],prkl]
        end
    end
    #Vijklchk = [Vpr[prodrev[i,j],prodrev[k,l]] for i=1:Nijkl, j=1:Nijkl, k=1:Nijkl, l=1:Nijkl]
    #@show norm(Vijklchk-Vijkl),"remove me"
    Vijkl        # cdagup cup cdagdn cdn
end


@compile_workload begin
    for iter=1:2
        orbs,~,~ = svd(randn(4,4))
        g,si = getgates(orbs;reportsign=true)
        #@show si, checkgates(g,orbs)
    end
end


end

