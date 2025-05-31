mutable struct DetChain{T}
    LDetsUp::Matrix{T}
    LDetsDn::Matrix{T}
    RDetsUp::Matrix{T}
    RDetsDn::Matrix{T}
    OinvsUp::Matrix{T}
    OinvsDn::Matrix{T}
    center::Int
end

function getLDets(x::DetChain, i::Int)
    if i > center
        error("Invalid index")
    end
    return x.LDetsUp[i], x.LDetsDn[i]
end

function getRDets(x::DetChain, i::Int)
    if i < center
        error("Invalid index")
    end
    return x.RDetsUp[i], x.RDetsDn[i]
end

function getOinv(x::DetChain, i::Int)
    if 
end

function getDets(x::DetChain, i::Int)
    LDetUp, LDetDn = getLDets(x, i)
    RDetUp, RDetDn = getRDets(x, i)
    return 
end
