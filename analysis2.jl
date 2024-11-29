using DelimitedFiles
using Statistics
using PyPlot

N = 4*4

function readE0(fname)
    res = Dict()
    open(fname,"r") do file
        for line in eachline(file)
            name, data = split(line)  # Process each line
            res[name] = parse(Float64, data) / N
        end
    end
    return res
end

function MCAnalysis(data, N_skip, plot=false)
    data = copy(data)[N_skip:end]
    @assert length(data) > 0
    inteval = 10
    vals, errs = Vector{Float64}(), Vector{Float64}()
    for i=2:inteval:length(data)
        tmp = data[1:i]
        val = mean(tmp)
        err = std(tmp) / sqrt(i)
        push!(vals, val)
        push!(errs, err)
    end
    if plot
        figure()
        N = length(vals)
        p = PyPlot.errorbar(1:N, vals, errs)
    end
    return vals, errs
end

function geten(fname, N_skip)
    println(fname)
    data = readdlm(fname)
    steps = data[:,1]
    Eks = data[:,2]
    EVs = data[:,3]
    signs = data[:,4]
    Eks, errEks = MCAnalysis(Eks, N_skip)
    EVs, errEVs = MCAnalysis(EVs, N_skip)
    signs, err_signs = MCAnalysis(signs, N_skip)

    return Eks, errEks, EVs, errEVs, signs, err_signs
end

function getDensity(fnames, N_skip)
    
end

function getAll(fnames, Ntaus, N_skip)
    Ekst, errEkst, EVst, errEVst, signst, err_signst = Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
    for fname in fnames
        Eks, errEks, EVs, errEVs, signs, err_signs = geten(fname, N_skip)
        push!(Ekst, Eks[end])
        push!(errEkst, errEks[end])
        push!(EVst, EVs[end])
        push!(errEVst, errEVs[end])
        push!(signst, signs[end])
        push!(err_signst, err_signs[end])
    end
    Ekst = Ekst ./ signst
    EVst = EVst ./ signst
    errEkst = divErr(Ekst, errEkst, signst, err_signst)
    errEVst = divErr(EVst, errEVst, signst, err_signst)

    Ekst = Ekst ./ N
    errEkst = errEkst ./ N
    EVst = EVst ./ N
    errEVst = errEVst ./ N

    return Ekst, errEkst, EVst, errEVst, signst, err_signst
end

function divErr(A, errA, B, errB)
    tmpA = (errA ./ A).^2
    tmpB = (errB ./ B).^2
    tmp = tmpA .+ tmpB
    tmp = sqrt.(tmp)
    return abs.((A./B) .* tmp)
end

ion()

data0 = readE0("data/en0_2x2_nup2_ndn2.dat")
println(data0)

fnames = ["data/en2x2_nup2_ndn2_Nt10.dat","data/en2x2_nup2_ndn2_Nt20.dat","data/en2x2_nup2_ndn2_Nt30.dat"]#,"data/en2x2_nup2_ndn2_Nt40.dat","data/en2x2_nup2_ndn2_Nt50.dat","data/en2x2_nup2_ndn2_Nt60.dat","data/en2x2_nup2_ndn2_Nt70.dat"]#,"en4x4_Nt80.dat"]
Ntaus = [0,10,20,30]#,40,50,60,70]#,80]
taus = Ntaus .* 0.05

N_skip = 50
Ekst, errEkst, EVst, errEVst, signst, err_signst = getAll(fnames, Ntaus, N_skip)


insert!(Ekst, 1, data0["Ek0"])
insert!(EVst, 1, data0["EV0"])
insert!(errEkst, 1, 0.)
insert!(errEVst, 1, 0.)

Est = Ekst .+ EVst
errEst = errEkst .+ errEVst


fig1 = figure()
println(length(Ntaus), length(Ekst), length(errEkst))
PyPlot.errorbar(taus, Ekst, errEkst, marker="o")
axhline(data0["Ek_GS"], ls="--", c="k")
xlabel("\$\\tau\$", fontsize=18)
ylabel("\$E_k/N\$", fontsize=18)
tight_layout()
savefig("data/Ek_2x2_2up2dn.pdf")

fig2 = figure()
PyPlot.errorbar(taus, EVst, errEVst, marker="o")
axhline(data0["EV_GS"], ls="--", c="k")
xlabel("\$\\tau\$", fontsize=18)
ylabel("\$E_V/N\$", fontsize=18)
tight_layout()
savefig("data/EV_2x2_2up2dn.pdf")

fig3 = figure()
PyPlot.errorbar(taus, Est, errEst, marker="o")
axhline(data0["Ek_GS"]+data0["EV_GS"], ls="--", c="k")
xlabel("\$\\tau\$", fontsize=18)
ylabel("\$E/N\$", fontsize=18)
tight_layout()
savefig("data/E_2x2_2up2dn.pdf")

fig4 = figure()
PyPlot.errorbar(taus[2:end], signst, err_signst, marker="o")
xlabel("\$\\tau\$", fontsize=18)
ylabel("sign", fontsize=18)
tight_layout()
savefig("data/sign_2x2_2up2dn.pdf")


show()

