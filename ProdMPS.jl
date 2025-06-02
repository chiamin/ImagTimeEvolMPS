# A machine to compute the overlap of an MPS with a product state efficiently.
# Use makeProdMPS to generate a ProdMPS object.
# Use getOverlap! to compute the overlap with a product state.
using ITensors, ITensorMPS

# Return [[A1_emp, A1_up, A1_dn, A1_updn], [A2_tmp, ...], ...]
function getProdMPSTensors(mps::MPS)::Vector{Vector{ITensor}}
    iis = siteinds(mps)

    tens = Vector{Vector{ITensor}}()
    for i=1:length(mps)
        ii = iis[i]
        d = dim(ii)
        APs = Vector{ITensor}()
        for j=1:d
            p = onehot(ii=>j)
            AP = dag(p) * mps[i]
            push!(APs, AP)
        end
        push!(tens, APs)
    end
    return tens
end

mutable struct ProdMPS
    conf::Vector{Int64}
    Ls::Dict{Int,ITensor}
    Rs::Dict{Int,ITensor}
    iL::Int                 # Every sites <= iL have been computed and can be reused
    iR::Int                 # Every sites >= iR have been computed and can be reused
    const As::Vector{Vector{ITensor}}   # As[site][state]
end

function makeProdMPS(mps::MPS)::ProdMPS
    As = getProdMPSTensors(mps)
    N = length(mps)
    @assert length(As) == N

    Ls = Dict(0 => ITensor(1.))
    Rs = Dict(N+1 => ITensor(1.))

    conf = zeros(N)

    pMPS = ProdMPS(conf, Ls, Rs, 0, N+1, As)
    return pMPS
end

# Compute <pmps|conf>
function getOverlap!(pmps::ProdMPS, conf::Vector{Int64}; toRight=true)
    @assert length(conf) == length(pmps.conf)
    N = length(conf)
    L,R = ITensor(),ITensor()

    # Compare the new and the current configurations
    compare = (conf .!= pmps.conf)
    # Find the first and the last sites that have different states with the old configuration
    i1 = findfirst(compare)
    i2 = findlast(compare)
    # Update the left tensors upto i1-1
    for i=pmps.iL+1:i1-1
        pmps.Ls[i] = pmps.Ls[i-1] * pmps.As[i][conf[i]]
    end
    pmps.iL = i1-1
    # Update the right tensors upto i2+1
    for i=pmps.iR-1:-1:i2+1
        pmps.Rs[i] = pmps.Rs[i+1] * pmps.As[i][conf[i]]
    end
    pmps.iR = i2+1
    # Update the tensors between i1 and i2
    if toRight
        for i=i1:i2
            pmps.Ls[i] = pmps.Ls[i-1] * pmps.As[i][conf[i]]
        end
        pmps.iL = i2
    else
        for i=i2:-1:i1
            pmps.Rs[i] = pmps.Rs[i+1] * pmps.As[i][conf[i]]
        end
        pmps.iR = i1
    end
    # Compute the result
    res = pmps.Ls[pmps.iL] * pmps.Rs[pmps.iR]
    @assert order(res) == 0
    return res[1]
end
