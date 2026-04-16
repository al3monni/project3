include("data_load.jl")
include("utils.jl")
include("visualization.jl")
include("local_optima_network.jl")
include("plot_lon.jl")
include("genetic.jl")
include("nsga2.jl")

using YAML
using Statistics
using Dates

# ============== Load Configuration ==============

const CONFIG = YAML.load_file(joinpath(@__DIR__, "parameters.yaml"))

const DATASETS = CONFIG["datasets"]
const DATASET_NAME = CONFIG["landscape"]["dataset_name"]
const PENALTY = CONFIG["landscape"]["penalty"]

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
        local_optima = get_local_optima(landscape)

        f1 = plot_landscape(landscape)
        f2 = plot_landscape_polar(landscape)
        f3 = hinged_bitstring_map(landscape, local_optima)

        g, opt_index_map, basin_map = build_LON(landscape, local_optima, n_bits)
        basin_sizes = compute_basin_sizes(basin_map, local_optima)
        f4 = plot_lon(g, landscape, opt_index_map, basin_sizes)

        out_path = mkpath(joinpath(@__DIR__, "..", "img"))
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
    
    for i in 1:n_runs
        println("  Run $i / $n_runs")
        histories[i], results[i] = algorithm(landscape, pop_size, k_max; params...)  # -> [5, generations], EvoLP.Result
    end
    
    # Compute averaged history across runs
    hist_stack = cat(histories...; dims=3)     # (5, generations, n_runs)
    avg_history = mean(hist_stack; dims=3)     # (5, generations, 1)
    avg_history = dropdims(avg_history; dims=3)  # (5, generations)
    
    # Compute statistics across runs
    bests = [r.fxstar for r in results]
    avg_best = mean(bests)
    std_best = std(bests)
    min_best = minimum(bests)
    max_best = maximum(bests)

    # Print summary statistics
    println("  Summary across $n_runs runs:")
    println("    Mean best fitness: $(round(avg_best, digits=4))")
    println("    Std best fitness:  $(round(std_best, digits=4))")
    println("    Min best fitness:  $(round(min_best, digits=4))")
    println("    Max best fitness:  $(round(max_best, digits=4))")

    return avg_history, avg_best, std_best, min_best, max_best
end

function main()

    # run_visualizations()

    mkpath(OUTPUT_DIR)
    cp(joinpath(@__DIR__, "parameters.yaml"), joinpath(OUTPUT_DIR, "parameters.yaml"))

    println("Output will be saved to: $OUTPUT_DIR")

    all_results = Dict()

    for dataset in DATASETS
        println("\n" * "="^60)
        println("EXPERIMENT: $dataset")
        println("="^60)

        landscape = load_landscape(dataset)
        println("Loaded landscape with $(length(landscape)) points")

        dataset_results = Dict()

        # Run GA
        println("\nRunning GA...")
        dataset_results["GA"] = run(landscape, GA!, POPSIZE, GENERATIONS, GA_PARAMS, N_RUNS)

        # Run PSO
        println("\nRunning PSO...")
        dataset_results["PSO"] = run(landscape, PSO!, POPSIZE, GENERATIONS, PSO_PARAMS, N_RUNS)

        # # Run NSGA2
        # println("\nRunning NSGA2...")
        # dataset_results["NSGA2"] = run(landscape, NSGA2!, POPSIZE, GENERATIONS, NSGA2_PARAMS, N_RUNS)

        all_results[dataset] = dataset_results

        # # Save results to CSV
        # run_data = []
        # for (algo, data) in dataset_results
        #     avg_history, avg_best, std_best, min_best, max_best = data
        #     fitness = avg_best  # or min_best depending on the context

        #     push!(run_data, (
        #         algorithm = algo,
        #         best_fitness = fitness
        #     ))
        # end
    end

    println("Output saved to: $OUTPUT_DIR")

end

main()