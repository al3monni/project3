using Combinatorics
using Printf
using HDF5
using Statistics

# ===========================================================
# DATA LOADING AND FITNESS CALCULATION
# ===========================================================

mutable struct Landscape
    name::String
    accuracies::Vector{Float32}
    fitnesses::Vector{Float32}
    n_features::Int
end

# ============= Landscape and Fitness Functions =============

function load_landscape(filename::String)

    if filename == "triangle" || filename == "asymmetric"
        return triangle_landscape(filename)
    end

    n_features = CONFIG["datasets"][filename]["n_features"]
    landscape = Landscape(filename, Float32[], Float32[], n_features)

    # obtain a reference for the hdf5 file 
    filepath = joinpath("train", filename)
    if !isfile(filepath)
        filepath = joinpath("test", filename)
        if !isfile(filepath)
            error("File $filename not found in train/ or test/ directories.")
        end
    end

    h5open(filepath, "r") do f

        # select the "accuracies" dataset
        data = read(f[DATASET_NAME])


        # compute the mean of each row (raw accuracies)
        accuracies = vec(mean(data, dims=2))
        landscape.accuracies = accuracies

        # compute the fitnesses with penalty
        init_fitnesses!(landscape)
    end

    return landscape
end

function init_fitnesses!(landscape::Landscape)

    n = length(landscape.accuracies)
    fitnesses = Vector{Float32}(undef, n)

    @inbounds for x in 1:n
        penalty = PENALTY * count_ones(x) / landscape.n_features
        fitnesses[x] = landscape.accuracies[x] - penalty
    end

    landscape.fitnesses = fitnesses
end

function fitness(x::Integer, landscape::Landscape)

    n = length(landscape.accuracies)

    # handle out-of-bounds cases
    if x == 0 || x > n
        return 0
    end

    return landscape.fitnesses[x]
end

function accuracy(x::Integer, landscape::Landscape)

    n = length(landscape.accuracies)

    # handle out-of-bounds cases
    if x == 0 || x > n
        return 0
    end

    return landscape.accuracies[x]
end

# ==================== Triangle Function ====================

function triangle_function(b::Integer, m::Float64, s::Integer)

    r = abs(b)
    t = mod(r, 2s)

    if t <= s
        return m * t
    else
        return m * (2s - t)
    end
end

function asymmetric_triangle_function(b::Integer, m::Float64, s::Integer)
    
    if b < 31
        return triangle_function(b, m, s)
    else
        return triangle_function(b, 6.0, s)
    end
    
end

function triangle_landscape(filename::String)

    if filename == "triangle"
        triangle_function_to_use = triangle_function
    elseif filename == "asymmetric"
        triangle_function_to_use = asymmetric_triangle_function
    else
        error("Invalid filename for triangle landscape: $filename")
    end

    n = CONFIG["datasets"][filename]["n"]
    m = CONFIG["datasets"][filename]["m"]
    s = CONFIG["datasets"][filename]["s"]

    # preallocate the lookup table
    lookup = Vector{Float32}(undef, n)

    # precompute and store triangle fitness values
    @inbounds for x in 1:n                              # pay attention here
        lookup[x] = triangle_function_to_use(x, m, s)
    end

    println(lookup)

    return Landscape(filename, lookup, lookup, ceil(Int, log2(n)))
end

# =========================================================
# LOCAL OPTIMA AND NEIGHBORHOOD CALCULATIONS
# =========================================================

function get_local_optima(landscape::Vector{Float32}; k::Int=1)

    """ compute the set of local optima for a given landscape """
    
    n = length(landscape)
    bits = ceil(Int, log2(n))
    
    local_optima = Int[]
    
    for i in 1:n
        # Check if current fitness is >= all neighbors
        if all(landscape[i] >= landscape[j] for j in neighbors(i, bits; k=k))
            push!(local_optima, i)
        end
    end
    
    return local_optima
end

function neighbors(index::Int, n_bits::Int; k::Int=1)
    neigh = Int[]

    for d in 1:k
        for positions in combinations(0:n_bits-1, d)
            mask = 0
            for i in positions
                mask |= (1 << i)
            end

            neighbor = index ⊻ mask

            if neighbor != 0
                push!(neigh, neighbor)
            end
        end
    end

    return neigh
end

# =========================================================
# MISCELLANEOUS UTILITY FUNCTIONS
# =========================================================

function to_bitstring(x::Int, n_bits::Int)
    s = string(x, base=2)
    return lpad(s, n_bits, '0')
end

function next_run_dir(base::String)
    # ensure base directory exists
    isdir(base) || mkpath(base)

    # list existing runs
    dirs = readdir(base)

    # extract numbers from "runX"
    runs = Int[]
    for d in dirs
        m = match(r"^run(\d+)$", d)
        if m !== nothing
            push!(runs, parse(Int, m.captures[1]))
        end
    end

    next_id = isempty(runs) ? 1 : maximum(runs) + 1

    run_dir = joinpath(base, @sprintf("run%d", next_id))
    mkpath(run_dir)

    return run_dir
end

function polar_coordinates(coordinates; base_radius = 0.1, radial_scale = 0.4)

    n = length(coordinates)
    f = Float64.(coordinates)

    # Normalize fitness to [0, 1]
    fmin = minimum(f)
    fmax = maximum(f)

    fnorm = fmax == fmin ? fill(0.5, n) : (f .- fmin) ./ (fmax - fmin)

    # Radius = base circle + amplified variation
    r = base_radius .+ radial_scale .* fnorm

    θ = range(0, 2π, length = n + 1)[1:end-1]

    x = r .* cos.(θ)
    y = r .* sin.(θ)

    return x, y
end

function save_results(algorithm::String, landscape::Landscape, file::String, data)
    history, avg_best, std_best, min_best, max_best, pareto_front = data

    dataset_name = split(landscape.name, ".")[1]

    # History graph
    evolution_plot = plot_evolution(history, landscape, algorithm)
    save(joinpath(OUTPUT_DIR, "$(dataset_name)_$(algorithm)_evolution.png"), evolution_plot)

    # Entropy graph
    entropy_plot = plot_entropy(history, "$algorithm on $dataset_name")
    save(joinpath(OUTPUT_DIR, "$(dataset_name)_$(algorithm)_entropy.png"), entropy_plot)

    # Pareto front graph (for NSGA2)
    if pareto_front !== nothing
        pareto_plot = plot_pareto_front(pareto_front, "Pareto Front on $dataset_name")
        save(joinpath(OUTPUT_DIR, "$(dataset_name)_$(algorithm)_pareto.png"), pareto_plot)
    end

    # Append summary other stats to output file
    open(file, "a") do io
        println(io, "$algorithm,$avg_best,$std_best,$min_best,$max_best")
    end

end

function entropy(population::Vector{BitVector})
    n = length(population)
    if n == 0
        return 0.0
    end

    n_bits = length(population[1])
    bit_counts = zeros(Int, n_bits)

    for individual in population
        for j in 1:n_bits
            bit_counts[j] += individual[j] ? 1 : 0
        end
    end

    ent = 0.0
    for count in bit_counts
        p = count / n
        if p > 0 && p < 1
            ent -= p * log2(p) + (1 - p) * log2(1 - p)
        end
    end

    return ent
end

function bitvector_to_index(bits::BitVector)
    value = 0
    for b in bits
        value = (value << 1) | (b ? 1 : 0)
    end
    return value
end