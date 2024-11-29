function binaryNum_MPS(N::Int)
    function makeA(n)
        A = zeros(2,2,2)
        A[1,:,1] = A[2,:,2] = [1.0, 1.0]
        A[2,:,1] = [0.0, 2^(n-1)]
        return A
    end

    mps = []
    for i=1:N
        A = makeA(i)
        if i == 1
            A = A[2,:,:]
        elseif i == N
            A = A[:,:,1]
        end
        push!(mps, A)
    end

    return mps
end

function getValue(mps, conf::Vector{Int64})::Number
    @assert size(mps) == size(conf)
    N = size(mps,1)
    v = mps[1][conf[1],:]'
    for i=2:N-1
        v = v * mps[i][:,conf[i],:]
    end
    v = v * mps[N][:,conf[N]]
    return v
end

function getBinaryNum(conf::Vector{Int64})::Int
    bnum = reverse(conf .- 1)
    bstr = join(bnum)
    val = parse(Int, bstr, base=2)
    return val
end
