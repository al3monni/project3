using Combinatorics
using Printf
using HDF5
using Statistics
using Base.Threads

# ===========================================================
# DATA LOADING AND FITNESS CALCULATION
# ===========================================================

struct Landscape
    name        # string
    accuracies  # vector of numbers
    fitnesses   # vector of numbers
    n_features  # int
end

# The 32-value lookup (indexed by number of active bits: 0 to 31)
const TRIANGLE_ASYMMETRIC_TABLE = UInt8[
    0, 1, 2, 3, 4, 5, 4, 3, 2, 1,   # 0-9  active bits
    0, 1, 2, 3, 4, 5, 4, 3, 2, 1,   # 10-19
    0, 1, 2, 3, 4, 5, 4, 3, 2, 1,   # 20-29
    0, 6                             # 30-31 (exception: m=6 kicks in at 31)
]

# ============= Landscape and Fitness Functions =============

function load_landscape(filename::String)

    if filename == "triangle" || filename == "asymmetric"
        return triangle_landscape(filename)
        penalty = penalty * count_ones(x) / n_features
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
        data = read(f["accuracies"])
        
        # compute the mean of each row (raw accuracies)
        accuracies = vec(mean(data, dims=2))

        # compute the fitnesses with penalty
        penalty = (filename == "triangle" || filename == "asymmetric") ? CONFIG["landscape"]["$(CONFIG["datasets"][filename]["split"])_penalty"] / n_features : 0.0
        fitnesses = init_fitnesses(accuracies, n_features, penalty)

        return Landscape(filename, accuracies, fitnesses, n_features)
    end
end

function init_fitnesses(accuracies::Vector{Float32}, n_features::Int, penalty::Float64)

    n = length(accuracies)
    fitnesses = Vector{Float32}(undef, n)

    @inbounds for x in 1:n
        fitnesses[x] = accuracies[x] - (penalty * count_ones(x))
    end

    return fitnesses
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

function triangle(b::Integer, m::Integer, s::Integer)::Integer
    norm_b = count_ones(b)

    a = ceil(Integer, norm_b/s)

    if a % 2 == 1
        # g(b)
        if norm_b % s == 0
            return m * s
        else 
            return m * (norm_b % s)
        end
    else
        return m * (a * s - norm_b)
    end
end

# phenotype-level version: takes norm_b (= count of active bits) directly
# used for visualization — avoids the count_ones(k) trap in plot_triangle_phenotype
function triangle_phenotype(norm_b::Integer, m::Integer, s::Integer)::Integer
    a = ceil(Integer, norm_b / s)
    if a % 2 == 1
        return norm_b % s == 0 ? m * s : m * (norm_b % s)
    else
        return m * (a * s - norm_b)
    end
end

function asymmetric_triangle(b::Integer, m::Integer, s::Integer)

    #     if b < 31
    #         return triangle(b, m, s)
    #     else
    #         return triangle(b, 6, s)
    #     end

    active_bits = count_ones(b)
    return TRIANGLE_ASYMMETRIC_TABLE[active_bits + 1]
    
end

function triangle_landscape(filename::String)

    if filename == "triangle"
        triangle_function = triangle
    elseif filename == "asymmetric"
        triangle_function = asymmetric_triangle
    else
        error("Invalid filename for triangle landscape: $filename")
    end

    n = CONFIG["datasets"][filename]["n"]
    m = CONFIG["datasets"][filename]["m"]
    s = CONFIG["datasets"][filename]["s"]

    # Indices 1..2^n-1: same convention as real datasets (index 0 = no features, excluded).
    # Any n-bit XOR of a value in [1, 2^n-1] stays in [0, 2^n-1]; 0 is filtered in neighbors().
    size = 2^n - 1
    lookup = Vector{Float32}(undef, size)
    @threads for x in 1:size
        lookup[x] = Float32(triangle_function(x, m, s))
    end
    return Landscape(filename, lookup, lookup, n)
end

# =========================================================
# LOCAL OPTIMA AND NEIGHBORHOOD CALCULATIONS
# =========================================================

function get_local_optima(landscape::Landscape; k::Int=1)
    n = length(landscape.accuracies)
    bits = landscape.n_features
    local_optima = Int[]
    for i in 1:n
        neighborhood = neighbors(i, bits; k=k)
        if all(landscape.accuracies[i] >= landscape.accuracies[j] for j in neighborhood)
            push!(local_optima, i)
        end
    end
    return local_optima
end

function neighbors(index::Int, n_bits::Int; k::Int=1)
    # Initialize an empty array to store neighbor indices
    neigh = Int[]

    # Loop over Hamming distances from 1 up to k
    for d in 1:k
    # Generate all combinations of bit positions of size d
        for positions in combinations(0:n_bits-1, d)
            mask = 0
            # Build a bitmask with 1s in the selected positions
            for i in positions
                mask |= (1 << i)
            end

            # Flip the selected bits using XOR to get a neighbor
            neighbor = index ⊻ mask

            # Exclude the zero index (optional constraint)
            if neighbor != 0
                push!(neigh, neighbor)
            end
        end
    end

    # println("Neighbors of ", to_bitstring(index, n_bits)," (k=$k): ", to_bitstring.(neigh, n_bits))

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
    out_dir = dirname(file)  # same directory as the CSV results file

    evolution_plot = plot_evolution(history, landscape, algorithm)
    save(joinpath(out_dir, "$(dataset_name)_$(algorithm)_evolution.png"), evolution_plot)

    entropy_plot = plot_entropy(history, "$algorithm on $dataset_name")
    save(joinpath(out_dir, "$(dataset_name)_$(algorithm)_entropy.png"), entropy_plot)

    if pareto_front !== nothing
        pareto_plot = plot_pareto_front(pareto_front, "Pareto Front on $dataset_name")
        save(joinpath(out_dir, "$(dataset_name)_$(algorithm)_pareto.png"), pareto_plot)
    end

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