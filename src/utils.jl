function get_local_optima(landscape::Vector{Float32})

    """ compute the set of local optima for a given landscape """
    
    n = length(landscape)
    bits = ceil(Int, log2(n))
    
    local_optima = Int[]
    
    for i in 1:n
        # Check if current fitness is >= all neighbors
        if all(landscape[i] >= landscape[j] for j in neighbors(i, bits))
            push!(local_optima, i)
        end
    end
    
    return local_optima
end

function neighbors(index::Int, n_bits::Int)

    """ optimized function to generate the neighbours of a given individual """

    neigh = Int[]

    for i in 0:n_bits-1

        neighbor = index ⊻ (1 << i)  # flip bit i

        if neighbor != 0
            push!(neigh, neighbor)  
        end
    end

    return neigh
end

using Printf

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

function save_results(algorithm::String, file::String, data)
    history, avg_best, std_best, min_best, max_best = data

    dataset_name = split(split(file, "/")[end-1], "_")[1]  # extract dataset name from path

    # History graph
    evolution_plot = plot_evolution(history, "$algorithm on $dataset_name")
    save(joinpath(OUTPUT_DIR, "$(dataset_name)_$(algorithm)_evolution.png"), evolution_plot)

    # Entropy graph
    entropy_plot = plot_entropy(history, "$algorithm on $dataset_name")
    save(joinpath(OUTPUT_DIR, "$(dataset_name)_$(algorithm)_entropy.png"), entropy_plot)

    # Append summary other stats to output file
    open(file, "a") do io
        println(io, "$algorithm,$avg_best,$std_best,$min_best,$max_best")
    end

end