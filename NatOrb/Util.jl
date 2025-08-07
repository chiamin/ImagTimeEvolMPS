module Util	# Utilities
using LinearAlgebra, Arpack, CPUTime, SparseArrays, Printf, KrylovKit  #,LinearMaps

eye(n) = Matrix(1.0I,n,n)
krondelta(i,j) = (i == j ? 1.0 : 0.0)

function unitvec(i,n)
    res = zeros(n)
    res[i] = 1.0
    res
end

function eig(M::Matrix{Float64})
    F = eigen(Symmetric(M))
    F.values,F.vectors
end
function eig(M::Matrix{ComplexF64})
    F = eigen(Hermitian(M))
    F.values,F.vectors
end
eigsym(M) = eig(M)

function geneig(O,H;evalcut = 1.0e-10)          # Assume O and H are Hermitian
    oevals,oevecs = eigen(Hermitian(O))
    filt = filter(i->oevals[i]>evalcut,1:length(oevals))
    Q = oevecs[:,filt]
    Hnew = Q' * H * Q
    Onew = Q' * O * Q
    hevals,evecs = eigen(Hermitian(Hnew),Hermitian(Onew))
    evecs = Q * evecs
    hevals,evecs
end


function Enuc(Na,R,Z=1.0)
  E = 0.0
  for d = 1:Na-1              # d = distances  
    E += (Na-d)/abs(1. * R * d)
  end
  return E*Z*Z
end

function diagrho(rho)
    eval,evec = eigsym(-rho)
    for rr = 1:length(eval)
        sum(evec[:,rr]) < 0.0 && (evec[:,rr] *= -1.0)
    end
    -eval,evec
end

function doeigs(H,n_ev::Int,v_initial;verbose=true,maxiter=0,tol=1e-8)	# H may be dense or sparse, should be symmetric to good acc
    isspar = (typeof(H) != Array{Float64,2})
    Huse = H
    if isspar
        numnonzero = nnz(H)
        verbose && (@show numnonzero)
        if numnonzero > size(H,1)*size(H,2) * 0.05
            Huse = Matrix(H)
            verbose && println("using full matrix")
        end
    end
    n = size(H,1)
    maxiter == 0 && (maxiter = n)
    if n_ev > 1 && n < 2000
        evals,evecs = eigsym(Huse)
        return evals[1:n_ev],evecs[:,1:n_ev]
    end
    hf(p) = Huse * p
    #HM = LinearMap(hf, n; issymmetric=true)
    #d,V,nconv,niter,nmult,resid = Arpack.eigs(HM;nev=n_ev,which=:SR,ritzvec=true,
    #                                          v0=v_initial,maxiter,tol)
    vals,vecs,info = eigsolve(hf,v_initial,n_ev,:SR;ishermitian=true,tol=1e-8, verbosity=0,
                                                krylovdim=20,maxiter=100)

    #@show typeof(v_initial),typeof(d)
    #verbose && (@show nmult)
    return vals,vecs
end

function conditionsvd(A)
    nrows,ncols = size(A)
    u,d,v = 0,0,0
    if nrows > ncols
        evals,evecs = eig(A' * A)
        u,d,v = svd(A * evecs;alg=LinearAlgebra.QRIteration())
        v = evecs * v
    else
        evals,evecs = eig(A * A')
        u,d,v = svd(evecs' * A;alg=LinearAlgebra.QRIteration())
        u = evecs * u
    end
    return u,d,v
end

function catchsvd(A)
    u,d,v = 0,0,0
    try
        u,d,v = svd(A)
    catch exc
        try
            u,d,v = svd(A;alg=LinearAlgebra.QRIteration())
        catch
            u,d,v = conditionsvd(A)
        end
    end
    return u,d,v
end

function complement(Akeep,Aold)	# Within the basis of Aold, find the space orthogonal to Akeep
    if size(Akeep,1) != size(Aold,1)
        @show size(Akeep),size(Aold)
        error("Inconsistent sizes in complement")
    end
    if length(size(Akeep)) == 1
        Akeep = reshape(Akeep,size(Akeep,1),1)
    end
    #u,d,v = svd(Akeep)
    n,k = size(Akeep)
    Ak = eye(n)
    Ak[:,1:k] = Akeep
    u = Matrix(qr(Ak).Q)[:,1:size(Akeep,2)]     #u now spans the space of Akeep 
    Asub = Aold - u * (u' * Aold)
    nret = size(Aold,2)-size(Akeep,2)
    try
        u,d,v = svd(Asub)
    catch exc
        println("Redoing svd in complement with eigsym")
        rho = -Asub * Asub'
        evals,evecs = eigsym(rho)
        return evecs[:,1:nret]
    end
    u[:,1:nret]
end

function blockformV(V,blocks,blockvecs)	# blocks must keep same number of states, Nperblock
    Nb = length(blocks)
    Nperblock = size(blockvecs[1],2)
    Vu = Array{Any,2}(undef,Nb,Nb)
    @inbounds for a=1:Nb, ap=1:Nb
	VV = V[blocks[a],blocks[ap]]
	na,nap = size(VV)
	UUa = Float64[blockvecs[a][i,m]*blockvecs[a][i,n] for i=1:na,m=1:Nperblock,n=1:Nperblock]
	UUap = Float64[blockvecs[ap][i,m]*blockvecs[ap][i,n] for i=1:nap,m=1:Nperblock,n=1:Nperblock]
	Ua = reshape(UUa,na,Nperblock^2)
	Uap = reshape(UUap,nap,Nperblock^2)
	Vu[a,ap] = reshape(Ua' * VV * Uap,Nperblock,Nperblock,Nperblock,Nperblock)
    end
    Vu
end

# V_{ijkl}^{nn'} -> cdag(i,n) cdag(j,n') c(k,n') c(l,n)
# V: nj,nj,nj,nj,Nb,Nb;   nj == Nperblock
function VblocktoV6(Vu)
    Nb = size(Vu,1)
    nj = size(Vu[1,1],1) 
    V = zeros(nj,nj,nj,nj,Nb,Nb)
    @inbounds for a=1:Nb, ap=1:Nb
	for i=1:nj,j=1:nj,k=1:nj,l=1:nj
	    V[i,k,l,j,a,ap] = Vu[a,ap][i,j,k,l]
	end
    end
    V
end

parseint(s) = parse(Int64,s)
parsefloat(s) = parse(Float64,s)

dorunout(a) = run(pipeline(`$(split(a))`))
dorunout(a,b) = run(pipeline(`$(split(a))`,stdout=b))
# Example: dorunout("$dbin/hfdmrg.jl 10 $R ","hf.res") to run hfdmrg.jl living in dbin, output hf.res

getmemory() = parsefloat(read(pipeline(`ps -p $(getpid()) -o rss`,`head -2`,`tail -1`),String))

function symmetrize!(m::Array{Float64,2})	# not usually needed; use eigsym
    n = size(m,1)
    for i=1:n, j=i:n
        m[i,j] = m[j,i] = 0.5 * (m[i,j]+m[j,i])
    end
end

manypush!(a,b,c...) = (push!(a,b); manypush!(c...))
manypush!(a,b) = push!(a,b)

mutable struct PrintRou
    str
end
import Base.show
Base.show(io::IO, z::PrintRou) = print(io, z.str)

orou3(x) = round(x,sigdigits=3)
#=
rou3(x::AbstractFloat) = lpad(replace((Printf.@sprintf "%7.3g" x),"e-0"=>"e-"),8)
rou3(x::Vector) = join(rou3.(x),", ")
rou3(x::Tuple) = join(rou3.(x),", ")
rou3(x) = x
=#
rrou3(x::AbstractFloat) = lpad(replace((Printf.@sprintf "%7.3g" x),"e-0"=>"e-"),8)
rrou3(x) = x
rou3(x::Vector) = PrintRou(join(rou3.(x),", "))
rou3(x::Tuple) = PrintRou(join(rou3.(x),", "))
rou3(x) = PrintRou(rrou3(x))

rrou5(x::AbstractFloat) = lpad(replace((Printf.@sprintf "%10.5g" x),"e-0"=>"e-"),11)
rrou5(x) = x
rou5(x::Vector) = PrintRou(join(rou5.(x),", "))
rou5(x::Tuple) = PrintRou(join(rou5.(x),", "))
rou5(x) = PrintRou(rrou5(x))

rrou8(x::AbstractFloat) = lpad(replace((Printf.@sprintf "%13.8g" x),"e-0"=>"e-"),14)
rrou8(x) = x
rou8(x::Vector) = PrintRou(join(rrou8.(x),", "))
rou8(x::Tuple) = PrintRou(join(rrou8.(x),", "))
rou8(x) = PrintRou(rrou8(x))

printfirst10(v) = printsp(rou3.(v[1:min(10,length(v))])...)
printfirst20(v) = printsp(rou3.(v[1:min(20,length(v))])...)

showmemory() = (println("Current Memory size: ",rou3(getmemory()/1024.0^2)," Gb"); flush(stdout))

cputime() = CPUtime_us() * 1e-6

function showcpu()
    sec = cputime()
    println("Current CPU time: ",sec)
    flush(stdout)
end


printlnf(a...) = (println(a...); flush(stdout))
printsp(a) = printlnf(a)
printsp(a,b...) = (print(a," "); printsp(b...))

printtab(a) = printlnf(a)
printtab(a,b...) = (print(a,"\t"); printtab(b...))

# to files:
printspf(fi,a) = (println(fi,a); flush(fi))
printspf(fi,a,b...) = (print(fi,a," "); printspf(fi,b...))
printtabf(fi,a) = (println(fi,a); flush(fi))
printtabf(fi,a,b...) = (print(fi,a,"\t"); printtabf(fi,b...))

function lastname(f)
    mend = match(r".*/(.*)$",f)
    mend == nothing ? f : mend.captures[1]
end

macro showfn(a)
    quote
        println(lastname($(string(__source__.file))),":",
                $(__source__.line)," ",$(string(a))," = ",$(esc(a)))
        flush(stdout)
    end
end
macro showf(a)
    quote
        println($(string(a))," = ",$(esc(a)))
        flush(stdout)
    end
end
macro showr(a)
    quote
	println($(string(a))," = ",rou3($(esc(a))))
        flush(stdout)
    end
end
macro showr(a::Tuple)
    quote
	print($(string(a))," = ",join(rou3.($(esc(a))),", "))
        flush(stdout)
    end
end

function findlastgreater(v,lim) #keeps at least one
    local i = 1
    while i < length(v) && abs(v[i+1]) > lim i += 1 end
    i
end

macro showfirst3(ev) quote
        printlnf($(string(ev))," = ", rou3($(esc(ev))[1:min(3,length($(esc(ev))))])...)
end end
macro showevals(ev) quote
        printlnf($(string(ev))," = ", rou3.($(esc(ev))[1:min(10,length($(esc(ev))))])...)
end end
macro showevalsend(ev) quote
        printlnf($(string(ev))," = ", rou3.($(esc(ev))[max(1,length($(esc(ev)))-9):end])...)
end end

function firstline(s,n=100)
    i = 2
    while i < length(s) && s[i:i] != "\n" && i < n
        i += 1
    end
    s[i:i] == "\n" && (i = i-1)
    s[1:i]
end


# only reports times greater than one second
macro timebig(ex) quote
        while false; end # compiler heuristic: compile this block (alter this if the heuristic changes)
        local elapsedtime = cputime()
        local val = $(esc(ex))
        local exx = $(string(ex))
        local exf = firstline(exx,20)
        local n=0
        elapsedtime = cputime() - elapsedtime
        if elapsedtime > 1.0
            printlnf(exf," took ",rou3(elapsedtime)," seconds")
        end
        val
end end

macro include(filename::AbstractString)
    path = joinpath(dirname(String(__source__.file)), filename)
    return esc(Meta.parse("quote; " * read(path, String) * "; end").args[1])
end
# usage  @include("file.jl")   evaluates in local scope, not global

atomnames = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar"]
#empirical atomic radii, in pm, converted to Bohr
atomicradii = [25, 120, 145, 105, 85, 70, 65, 60, 50, 160, 180, 150, 125, 110, 100, 100, 100, 71] / 53 

function status(s)
    f=open("Status","w")
    printlnf(f,s)
    println(s)
    close(f)
end

function findzero(f,amin,amax) 
    @assert f(amin)*f(amax) < 0.0
    ac = amin
    while amax-amin > 1.0e-8
        ac = 0.5 * (amin+amax)
        if f(amin)*f(ac) < 0.0
            amax = ac
        else
            amin = ac
        end
    end
    ac
end

# These two functions were written by chat gpt4 !!
function float_to_4string(x::Float64)
    if 0.0001 <= abs(x) < 9999
        return strip(rstrip(@sprintf("%.4f", x), '0'), '.')
    else
        return @sprintf("%.4e", x)
    end
end 

macro show4(var)
    return quote
        var_name = Symbol($(QuoteNode(var)))
        if typeof($(esc(var))) <: AbstractVector || typeof($(esc(var))) <: Tuple
            formatted_data = map($(esc(var))) do item
                if typeof(item) <: AbstractFloat
                    float_to_4string(item)
                else
                    string(item)
                end
            end
            if typeof($(esc(var))) <: AbstractVector
                println(var_name, " = [", join(formatted_data, ", "), "]")
            else
                println(var_name, " = (", join(formatted_data, ", "), ")")
            end
        elseif typeof($(esc(var))) <: AbstractFloat
            println(var_name, " = ", float_to_4string($(esc(var))))
        else
            println(var_name, " = ", $(esc(var)))
        end
    end
end

function axpy(vmps,coefs)
    ss = vmps[1] * coefs[1]
    for i=2:length(coefs)
        ss += vmps[i] * coefs[i]
    end
    ss
end
function linearfit(x,y,s)       # y versus x plot, sigma = s.  In dmrg, x=s=truncation err, y = energy
    xs2 = sum(x ./ s.^2)
    x2s2 = sum(x.^2 ./ s.^2)
    ys2 = sum(y ./ s.^2)
    xys2 = sum(x .* y ./ s.^2)
    s2 = sum(1.0 ./ s.^2)
    slope = (xs2*ys2-xys2*s2)/(xs2^2-x2s2*s2)
    intercept = (xys2 - slope * x2s2)/xs2
    denom  = x2s2*s2-xs2^2
    slopeerr = sqrt(s2/denom)
    intercepterr = sqrt(x2s2/denom)

    predicted = x .* slope .+ intercept
    dmrgerr = (minimum(en)-intercept)*0.2
    (intercept,slope,intercepterr,slopeerr,dmrgerr,predicted)
end

getdefaults(; kwargs...) = kwargs
axisdefaults = getdefaults(; xlabel="xlabel",ylabel="ylabel",xgridvisible=false,ygridvisible=false,
                     xtickalign=1,ytickalign=1,
                     xlabelsize=30,ylabelsize=30,
                     xticksize=18,yticksize=18,
                     xtickwidth=3,ytickwidth=3,
                     spinewidth=3,
                     xminorticksize=12,yminorticksize=12,
                     xticklabelsize=30,yticklabelsize=30)
#Usage: Axis(f[1,1];axisdefaults...)

#function standardaxis(f; xlabel="xlabel",ylabel="ylabel",xgridvisible=false,ygridvisible=false,
#                      xtickalign=1,ytickalign=1,kwargs...)
#    Axis(f;xlabel,ylabel,xgridvisible,ygridvisible,xtickalign,ytickalign,kwargs...)
#end
    


export  eye, unitvec, eigsym, eig, Enuc, diagrho, doeigs, conditionsvd, catchsvd, complement, parseint, parsefloat, 
        dorunout, symmetrize!, manypush!,rou3,orou3,rou5,rou8,getmemory,showmemory,cputime,showcpu, @showr,
        blockformV, VblocktoV6, printsp,@showf,lastname,printlnf,printfirst10,printfirst20, printspf,printtab,printtabf,
        findlastgreater,@showevals,@showevalsend,@timebig,firstline,@include,atomnames,atomicradii,
        @showfirst3, status, findzero,@show4, axpy, krondelta, linearfit,
        axisdefaults, geneig

end


