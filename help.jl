include("SampleMPSDet.jl")
include("SampleOcc.jl")
include("DetTools.jl")
include("Measure.jl")
using Combinatorics, LinearAlgebra, BitIntegers

function parse_value(s)
    s_lower = lowercase(s)
    if s_lower == "true"
        return true
    elseif s_lower == "false"
        return false
    end
    try
        return parse(Int, s)
    catch
        try
            return parse(Float64, s)
        catch
            return s
        end
    end
end

function read_params(filename)
    params = Dict{String, Any}()
    for line in eachline(filename)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue

        if occursin("=", stripped)
            keyval = split(stripped, "=", limit=2)
            key = strip(keyval[1])
            raw_values = split(strip(keyval[2]))
        else
            parts = split(stripped)
            length(parts) < 2 && continue
            key = parts[1]
            raw_values = parts[2:end]
        end

        values = parse_value.(raw_values)
        params[key] = length(values) == 1 ? values[1] : values
    end
    return params
end



function getMPSAmplitudes(mps::MPS)
    confs = []
    ampls = []
    for i1=1:4  
      for i2=1:4
        for i3=1:4        
          for i4=1:4
            conf = [i1,i2,i3,i4]
            Nup, Ndn = getParNums(conf)
            if Nup == 2 && Ndn == 2
                ampl = MPSOverlap(conf,mps)
                if ampl != 0.
                    push!(confs, conf)
                    push!(ampls, ampl)
                end
            end
          end
        end
      end
    end
    println(confs)
    println(ampls)
    return confs, ampls
end

function generate_product_configurations(N_sites::Int, N_up::Int, N_dn::Int)
    valid_configs = []

    local_states = [1, 2, 3, 4]  # 1: empty, 2: up, 3: down, 4: double

    # Reverse the order of site iteration to make site 1 vary slowest
    for config in Iterators.product(ntuple(_ -> local_states, N_sites)...)
        config_vec = collect(config)
        config_vec_reversed = reverse(config_vec)  # Reverse to match desired order

        n_up = count(x -> x == 2 || x == 4, config_vec_reversed)
        n_dn = count(x -> x == 3 || x == 4, config_vec_reversed)

        if n_up == N_up && n_dn == N_dn
            push!(valid_configs, config_vec_reversed)  # Store reversed config
        end
    end

    return valid_configs
end

function generate_all_fields(n)
    tuples = Iterators.product(fill((1, 2), n)...)
    return [collect(t) for t in tuples]  # convert each tuple to a vector
end

function print_state_amplitudes(psi::Vector{Float64}, basis::Vector{Tuple{Int, Int}}, N_sites::Int)
    for (idx, (up_mask, dn_mask)) in enumerate(basis)
        amplitude = psi[idx]
        config = Int[]
        for site in 0:N_sites-1  # bit positions: least significant bit = site 0
            n_up = (up_mask >> site) & 1
            n_dn = (dn_mask >> site) & 1
            if n_up == 0 && n_dn == 0
                push!(config, 1)  # empty
            elseif n_up == 1 && n_dn == 0
                push!(config, 2)  # up
            elseif n_up == 0 && n_dn == 1
                push!(config, 3)  # down
            elseif n_up == 1 && n_dn == 1
                push!(config, 4)  # doubly occupied
            end
        end
        println("$(config) $amplitude")
    end
end

function generate_basis(N_sites::Int, N_up::Int, N_down::Int)
    up_configs = collect(combinations(0:N_sites-1, N_up))
    down_configs = collect(combinations(0:N_sites-1, N_down))
    basis = Tuple{Int, Int}[]
    
    for up_sites in up_configs
        up_mask = sum(1 << i for i in up_sites)
        for down_sites in down_configs
            dn_mask = sum(1 << i for i in down_sites)
            push!(basis, (up_mask, dn_mask))
        end
    end
    return basis
end

function get_normalized_slater_amplitudes(phi_up::AbstractMatrix{T}, phi_down::AbstractMatrix{T}; cutoff::Float64 = 1e-14) where T<:Number
    M, N_up = size(phi_up)
    _, N_down = size(phi_down)
    
    basis = generate_basis(M, N_up, N_down)

    configurations = Vector{Vector{Int}}()
    amplitudes = Vector{T}()

    for (idx, (up_mask, dn_mask)) in enumerate(basis)
        up_sites = [site for site in 0:M-1 if (up_mask >> site) & 1 == 1]
        down_sites = [site for site in 0:M-1 if (dn_mask >> site) & 1 == 1]
        
        config = Int[]
        for site in 0:M-1
            n_up = (up_mask >> site) & 1
            n_dn = (dn_mask >> site) & 1
            if n_up == 0 && n_dn == 0
                push!(config, 1)
            elseif n_up == 1 && n_dn == 0
                push!(config, 2)
            elseif n_up == 0 && n_dn == 1
                push!(config, 3)
            elseif n_up == 1 && n_dn == 1
                push!(config, 4)
            end
        end

        total_amp = overlap(config, phi_up, phi_down)

        push!(configurations, config)
        push!(amplitudes, total_amp)
    end

    # Normalize amplitudes
    norm_val = sqrt(sum(abs2, amplitudes))
    amplitudes_normalized = amplitudes ./ norm_val

    # Apply cutoff
    amplitudes_cleaned = map(a -> abs(a) < cutoff ? zero(a) : a, amplitudes_normalized)

    # Print configurations and cleaned amplitudes
    for (config, amp) in zip(configurations, amplitudes_cleaned)
        println("$(config) $amp")
    end

    return configurations, amplitudes_cleaned
end

function getEnergy(phi_up, phi_dn, Hk, U)
    G_up = Greens_function(phi_up, phi_up)
    G_dn = Greens_function(phi_dn, phi_dn)

    Ek = kinetic_energy(G_up, G_dn, Hk)
    EV = potential_energy(G_up, G_dn, U)
    return Ek+EV
end

# Write psi to file
function writeMPS(psi, filename)
    confs, ampls = getMPSAmplitudes(psi)
    open(filename,"w") do file
        for i=1:length(confs)
            println(file,confs[i]," ",ampls[i])
        end
    end
end

"""
    compute_all_amplitudes(Φ_up::Matrix, Φ_dn::Matrix)

Compute wavefunction amplitudes for all real-space basis configurations.
Each configuration is a vector of Ints: 1 (empty), 2 (up), 3 (down), 4 (doubly occupied)
"""
function compute_all_amplitudes(Φ_up::Matrix, Φ_dn::Matrix)
    L = size(Φ_up, 1)
    N_up = size(Φ_up, 2)
    N_dn = size(Φ_dn, 2)

    configs = Vector{Vector{Int}}()
    amplitudes = Float64[]

    for up_sites in combinations(1:L, N_up)
        for dn_sites in combinations(1:L, N_dn)
            # Build configuration
            config = fill(1, L)
            for i in up_sites
                config[i] = 2
            end
            for i in dn_sites
                if config[i] == 2
                    config[i] = 4  # doubly occupied
                else
                    config[i] = 3
                end
            end

            # Extract rows for determinant calculation
            M_up = Φ_up[collect(up_sites), :]
            M_dn = Φ_dn[collect(dn_sites), :]
            amp = det(M_up) * det(M_dn)

            push!(configs, config)
            push!(amplitudes, amp)
        end
    end

    return configs, amplitudes
end

function writeDet(Φ_up::Matrix, Φ_dn::Matrix, filename::String)
    configs, amplitudes = compute_all_amplitudes(Φ_up, Φ_dn)

    open(filename, "w") do io
        for (conf, amp) in zip(configs, amplitudes)
            if abs(amp) > 1e-12  # tolerance for numerical zero
                println(io, string(conf), " ", amp)
            end
        end
    end
end

