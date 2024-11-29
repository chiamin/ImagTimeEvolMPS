using Random

function sampleProb(Ps::Vector{Float64})::Int
    ws = [Ps[1]]
    for i=2:length(Ps)
        w = ws[end] + Ps[i]
        push!(ws, w)
    end

    rnum = rand() * ws[end]
    for i=1:length(ws)
        if rnum < ws[i]
            return i
        end
    end

    throw(ErrorException("Cannot find a choice"))
end
