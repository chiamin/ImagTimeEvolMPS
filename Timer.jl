using DataStructures

if !@isdefined timer
    const timer = DefaultDict{String,Float64}(0.)
end
