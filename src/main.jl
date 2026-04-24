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
const TRAIN = [dataset for dataset in keys(DATASETS) if DATASETS[dataset]["split"] == "train"]
const TEST = [dataset for dataset in keys(DATASETS) if DATASETS[dataset]["split"] == "test"]
const SYNTHETIC = [dataset for dataset in keys(DATASETS) if DATASETS[dataset]["split"] == "synthetic"]
const NEIGHBORHOOD_SIZE = CONFIG["visualization"]["neighborhood_size"]

const N_RUNS = CONFIG["experiment"]["n_runs"]
const POPSIZE = CONFIG["experiment"]["popsize"]
const GENERATIONS = CONFIG["experiment"]["generations"]
const RUN_OUTPUT_DIR = next_run_dir(CONFIG["experiment"]["output_dir"])

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

function run_visualizations(datasets=keys(DATASETS))

    println("\nRunning landscape visualizations...")

    for dataset in datasets
        for k in 1:NEIGHBORHOOD_SIZE

            # k = 1
            # dataset = "triangle"
            
            # if dataset == "triangle"
            #     landscape = triangle_landscape(SYNTHETIC["n"]; m=SYNTHETIC["m"], s=SYNTHETIC["s"])
            #     triangle = true
            # else
            #     landscape = load_landscape(dataset)
            #     triangle = false
            # end
            
            landscape = load_landscape(dataset)

            n_bits = landscape.n_features
            
            local_optima = get_local_optima(landscape; k=k)

            f1 = plot_landscape(landscape, local_optima; show_points = triangle)
            #f2 = plot_landscape_polar(landscape)
            f3 = hinged_bitstring_map(landscape, local_optima)

            # ==================== LON ====================
            
            # Build LON
            g, opt_index_map, basin_map = build_LON(landscape, local_optima, n_bits; k=NEIGHBORHOOD_SIZE)

            # Compute basin sizes
            basin_sizes = compute_basin_sizes(basin_map, local_optima)

            # Export LON
            #export_LON(landscape, g, opt_index_map, basin_sizes)

            # Plot LON
            f4, _ = plot_lon(g, landscape, opt_index_map, basin_sizes)

            display(f1)
            #display(f2)
            #display(f3)
            #display(f4)

            base_path = joinpath(@__DIR__, "..", "img")

            dataset_short = split(dataset, ".")[1]
            out_path = joinpath(base_path, dataset_short, "k_$(k)")

            mkpath(out_path)

            #save(joinpath(out_path, "$(dataset_short)_k$(k)_2Dlandscape.png"), f1)
            #save(joinpath(out_path,"$(dataset_short)_k$(k)_landscape_polar.png", f2)
            #save(joinpath(out_path, "$(dataset_short)_k$(k)_hinged_bitstring_map.png"), f3)
            #save(joinpath(out_path, "$(dataset_short)_k$(k)_lon.png"), f4)

            println("  Saved visualizations for $dataset")
        end
    end
end

# ============== Run Experiment ==============

function run(landscape, algorithm, pop_size, k_max, f, params, n_runs) # -> average history across runs + statistics
    
    histories = Vector{Any}(undef, n_runs) # -> [history1, history2, ...] where history_i is [5, generations]
    results = Vector{Any}(undef, n_runs)
    pareto_fronts = Vector{Any}(undef, n_runs)

     for i in 1:n_runs
        println("  Run $i / $n_runs")
        histories[i], results[i], pareto_fronts[i] = algorithm(landscape, pop_size, k_max, f; params...) # -> [5, generations], EvoLP.Result, pareto_front
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
    std_best = isnan(std(bests)) ? 0.0 : std(bests)
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

function run_experiments(datasets=keys(DATASETS))

    mkpath(RUN_OUTPUT_DIR)
    cp(joinpath(@__DIR__, "parameters.yaml"), joinpath(RUN_OUTPUT_DIR, "parameters.yaml"))

    println("Output will be saved to: $RUN_OUTPUT_DIR")

    for dataset in datasets
        println("\n" * "="^60)
        println("EXPERIMENT: $dataset")
        println("="^60)

        landscape = load_landscape(dataset)
        println("Loaded landscape with $(length(landscape.accuracies)) points")

        # Initialize results file for this dataset
        output_path = joinpath(RUN_OUTPUT_DIR, "$(split(dataset, ".")[1])_best_fitness.csv")
        open(output_path, "w") do io
            println(io, "algorithm,mean_best,std_best,min_best,max_best")
        end

        # Run GA
        println("\nRunning GA...")
        results = run(landscape, GA!, POPSIZE, GENERATIONS, fitness, GA_PARAMS, N_RUNS)
        save_results("GA", landscape, output_path, results)

        # Run PSO
        println("\nRunning PSO...")
        results = run(landscape, PSO!, POPSIZE, GENERATIONS, fitness, PSO_PARAMS, N_RUNS)
        save_results("PSO", landscape, output_path, results)

        # Run NSGA2
        println("\nRunning NSGA2...")
        results = run(landscape, NSGA2!, POPSIZE, GENERATIONS, evaluate_multiobjective, NSGA2_PARAMS, N_RUNS)
        save_results("NSGA2", landscape, output_path, results)

    end

    println("Output saved to: $RUN_OUTPUT_DIR")
end

function main()

    # 1. Visualization of landscapes
    run_visualizations(TRAIN)

    # 2. Run experiments (data on algorithms' average performance across runs)
    # run_experiments(TRAIN + triangle)
    
    # 2a. Visualizeion of algorithm behavior on landscapes (single run, ...)
    # run_behavior_visualizations(TRAIN)

    # 2b. Visualization of algorithm behavior on synthetic landscapes (single run, population distribution, ...)
    # run_behavior_visualizations(triangle)

    # 4. Visualization of test landscapes
    # run_visualizations(TEST + asymmetric)

    # 5. Run algorithms on test landscapes
    # run_experiments(TEST + asymmetric)

    # 5a. Visualizeion of algorithm behavior on test landscapes (single run, ...)
    # run_behavior_visualizations(TEST )

    # 5b. Visualization of algorithm behavior on synthetic landscapes (single run, population distribution, ...)
    # run_behavior_visualizations(asymmetric)

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

    f1 = plot_landscape_with_path(landscape, best_path, local_optima) # DEBUG OK - poor visualization due to 2D projection

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

main()