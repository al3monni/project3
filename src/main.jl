include("utils.jl")
include("visualization.jl")
include("genetic.jl")
include("nsga2.jl")
include("algorithm_behavior.jl")

using YAML
using Statistics
using Dates

# ============== Load Configuration ==============

const CONFIG = YAML.load_file(joinpath(@__DIR__, "parameters.yaml"))

const DATASETS = CONFIG["datasets"]
const DATASET_NAME = CONFIG["landscape"]["dataset_name"]
const PENALTY = CONFIG["landscape"]["penalty"]
const NEIGHBORHOOD_SIZE = CONFIG["neighborhood"]["size"]

const N_RUNS = CONFIG["experiment"]["n_runs"]
const POPSIZE = CONFIG["experiment"]["popsize"]
const GENERATIONS = CONFIG["experiment"]["generations"]
const OUTPUT_DIR = next_run_dir(CONFIG["experiment"]["output_dir"])

const GA_PARAMS = Dict(
    :pc => CONFIG["algorithms"]["GA"]["pc"],
    :pm => CONFIG["algorithms"]["GA"]["pm"]
    )
const PSO_PARAMS = Dict(
    :w => CONFIG["algorithms"]["PSO"]["w"],
    :c1 => CONFIG["algorithms"]["PSO"]["c1"],
    :c2 => CONFIG["algorithms"]["PSO"]["c2"]
    )
const NSGA2_PARAMS = Dict(
    :pc => CONFIG["algorithms"]["NSGA2"]["pc"],
    :pm => CONFIG["algorithms"]["NSGA2"]["pm"]
    )
    
# ============== Visualizations ==============

function run_visualizations()

    println("\nRunning landscape visualizations...")

    for dataset in DATASETS

        landscape = load_landscape(dataset)
        n = length(landscape)
        n_bits = ceil(Int, log2(n))
        local_optima = get_local_optima(landscape; k=NEIGHBORHOOD_SIZE)

        f1 = plot_landscape(landscape)
        f2 = plot_landscape_polar(landscape)
        f3 = hinged_bitstring_map(landscape, local_optima)

        # ==================== LON ====================
        
        # Build LON
        g, opt_index_map, basin_map = build_LON(landscape, local_optima, n_bits; k=NEIGHBORHOOD_SIZE)

        # Compute basin sizes
        basin_sizes = compute_basin_sizes(basin_map, local_optima)

        # Export LON
        #export_LON(landscape, g, opt_index_map, basin_sizes)

        # Plot LON
        f4 = plot_lon(g, landscape, opt_index_map, basin_sizes)

        #display(f1)
        #display(f2)
        #display(f3)
        #display(f4)

        img = "img$k"
        out_path = joinpath(@__DIR__, "..", img)
        mkpath(out_path)

        dataset_short = split(dataset, ".")[1]

        save("$out_path/$(dataset_short)_landscape.png", f1)
        save("$out_path/$(dataset_short)_landscape_polar.png", f2)
        save("$out_path/$(dataset_short)_hinged_bitstring_map.png", f3)
        save("$out_path/$(dataset_short)_lon.png", f4)

        println("  Saved visualizations for $dataset")
    end
end

# ============== Run Experiment ==============

function run(landscape, algorithm, pop_size, k_max, params, n_runs) # -> average history across runs + statistics
    
    histories = Vector{Any}(undef, n_runs) # -> [history1, history2, ...] where history_i is [5, generations]
    results = Vector{Any}(undef, n_runs)
    pareto_fronts = Vector{Any}(undef, n_runs)

     for i in 1:n_runs
        println("  Run $i / $n_runs")
        histories[i], results[i], pareto_fronts[i] = algorithm(landscape, pop_size, k_max; params...) # -> [5, generations], EvoLP.Result, pareto_front
    end
    
    # Compute averaged history across runs
    hist_stack = cat(histories...; dims=3)     # (5, generations, n_runs)
    avg_history = mean(hist_stack; dims=3)     # (5, generations, 1)
    avg_history = dropdims(avg_history; dims=3)  # (5, generations)

    # Compute averaged pareto front across runs (for NSGA2)
    if !isempty(pareto_fronts) && pareto_fronts[1] !== nothing
        all_solutions = vcat(pareto_fronts...)
        unique_solutions = unique(all_solutions)
        pareto_front = pareto_filter(unique_solutions)
    else
        pareto_front = nothing
    end
    
    # Compute statistics across runs
    bests = [r.fxstar for r in results]
    avg_best = mean(bests)
    std_best = std(bests)
    min_best = minimum(bests)
    max_best = maximum(bests)

    # Print summary statistics
    println()
    println("Summary across $n_runs runs:")
    println("  Mean best fitness: $(round(avg_best, digits=4))")
    println("  Std best fitness:  $(round(std_best, digits=4))")
    println("  Min best fitness:  $(round(min_best, digits=4))")
    println("  Max best fitness:  $(round(max_best, digits=4))")
    println()

    return avg_history, avg_best, std_best, min_best, max_best, pareto_front
end

function main()

    # run_visualizations()

    mkpath(OUTPUT_DIR)
    cp(joinpath(@__DIR__, "parameters.yaml"), joinpath(OUTPUT_DIR, "parameters.yaml"))

    println("Output will be saved to: $OUTPUT_DIR")

    for dataset in keys(DATASETS)
        println("\n" * "="^60)
        println("EXPERIMENT: $dataset")
        println("="^60)

        landscape = load_landscape(dataset)
        println("Loaded landscape with $(length(landscape)) points")
        println("Max: $(maximum(landscape)), Min: $(minimum(landscape))")

        # Initialize results file for this dataset
        output_path = joinpath(OUTPUT_DIR, "$(split(dataset, ".")[1])_best_fitness.csv")
        open(output_path, "w") do io
            println(io, "algorithm,mean_best,std_best,min_best,max_best")
        end

        # Run GA
        println("\nRunning GA...")
        results = run(landscape, GA!, POPSIZE, GENERATIONS, GA_PARAMS, N_RUNS)
        save_results("GA", dataset, output_path, results)

        # Run PSO
        println("\nRunning PSO...")
        results = run(landscape, PSO!, POPSIZE, GENERATIONS, PSO_PARAMS, N_RUNS)
        save_results("PSO", dataset, output_path, results)

        # Run NSGA2
        println("\nRunning NSGA2...")
        results = run(landscape, NSGA2!, POPSIZE, GENERATIONS, NSGA2_PARAMS, N_RUNS)
        save_results("NSGA2", dataset, output_path, results)

    end

    println("Output saved to: $OUTPUT_DIR")

end

# ============== Test Behavior ===============

function test_behavior()

    # extract dataset
    dataset = keys(DATASETS)[1]

    # load landscape
    landscape = load_landscape(dataset)

    # run GA and get history
    history, _ = GA!(landscape, 10, 200)

    # extract best individual from history
    best_path = Int.(history[6, :])

    n = length(landscape)
    n_bits = ceil(Int, log2(n))

    # compute the local optima
    local_optima = get_local_optima(landscape; k=NEIGHBORHOOD_SIZE)

    # =================== Plots ===================

    f1 = plot_landscape_with_path(landscape, best_path) # DEBUG OK - poor visualization due to 2D projection

    f2 = plot_landscape_polar_with_path(landscape, best_path)

    f3 = plot_hinged_map_with_path(landscape, best_path, local_optima)

    # ==================== LON ====================

    # Build LON
    g, opt_index_map, basin_map = build_LON(landscape, local_optima, n_bits; k=NEIGHBORHOOD_SIZE)

    # Compute basin sizes
    basin_sizes = compute_basin_sizes(basin_map, local_optima)
 
    # Export LON
    # export_LON(landscape, g, opt_index_map, basin_sizes)

    # Plot LON
    f4 = plot_lon_with_path(g, landscape, opt_index_map, basin_map, basin_sizes, best_path)

    # =================== Saving ==================

    dir = "img_behavior_test"
    out_path = joinpath(@__DIR__, "..", dir)
    mkpath(out_path)

    dataset_short = split(dataset, ".")[1]

    save("$out_path/$(dataset_short)_landscape.png", f1)
    save("$out_path/$(dataset_short)_landscape_polar.png", f2)
    save("$out_path/$(dataset_short)_hinged_bitstring_map.png", f3)
    save("$out_path/$(dataset_short)_lon.png", f4)

    display(f1)
    display(f2)
    display(f3)
    display(f4)

end

# test_behavior()

main()