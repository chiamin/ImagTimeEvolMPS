if !@isdefined timer
    const timer = Dict{String,Float64}()
end

if !@isdefined startTime
    const startTime = Dict{String,Float64}()
end

function tstart(name::String)
    #@assert !haskey(startTime, name)
    startTime[name] = time_ns()
end

function tend(name::String)
    if haskey(timer, name)
        timer[name] += (time_ns() - startTime[name]) / 1e9
    else
        timer[name] = (time_ns() - startTime[name]) / 1e9
    end
    # Remove name in startTime
    pop!(startTime, name)
end

function treset()
    empty!(timer)
    empty!(startTime)
end
