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
        dataset_short = split(dataset, ".")[1]
        landscape = load_landscape(dataset)

        # --- Synthetic landscapes: phenotypic view ---
        if dataset == "triangle" || dataset == "asymmetric"
            out_path = joinpath(@__DIR__, "..", "img", dataset_short)
            mkpath(out_path)

            f = plot_triangle_phenotype(dataset)
            save(joinpath(out_path, "$(dataset_short)_phenotype.png"), f)
            println("  Saved phenotype plot for $dataset")
            continue
        end

        # --- Real feature-selection landscapes: full structural analysis ---
        n_bits = landscape.n_features

        for k in 1:NEIGHBORHOOD_SIZE
            local_optima = get_local_optima(landscape; k=k)

            f1 = plot_landscape(landscape.fitnesses, local_optima)
            f3 = hinged_bitstring_map(landscape.fitnesses, local_optima)

            g, opt_index_map, basin_map = build_LON(landscape.fitnesses, local_optima, n_bits; k=k)
            basin_sizes = compute_basin_sizes(basin_map, local_optima)
            f4, _ = plot_lon(g, landscape.fitnesses, opt_index_map, basin_sizes)

            out_path = joinpath(@__DIR__, "..", "img", dataset_short, "k_$(k)")
            mkpath(out_path)

            save(joinpath(out_path, "$(dataset_short)_k$(k)_2Dlandscape.png"), f1)
            save(joinpath(out_path, "$(dataset_short)_k$(k)_hinged_bitstring_map.png"), f3)
            save(joinpath(out_path, "$(dataset_short)_k$(k)_lon.png"), f4)

            println("  Saved visualizations for $dataset (k=$k)")
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

    # --- Training phase ---
    run_visualizations(TRAIN)
    run_experiments(TRAIN)

    # Optional: algorithm behaviour figures + GIFs
    run_behavior_visualizations(TRAIN)

    # --- Test phase (uncomment when ready) ---
    # run_visualizations(TEST)
    # run_experiments(TEST)

end

# ============== Algorithm Behaviour Visualizations ==============

# Runs one instance of each algorithm on a dataset and saves:
#   - A static multi-panel behaviour figure (PNG, good for report)
#   - A GIF animation of the best-individual trajectory (good for presentation)
#     Set make_gif=false to skip (requires FFMPEG).

function run_behavior_visualizations(
    datasets = TRAIN;
    algorithms = [("GA", GA!, fitness, GA_PARAMS),
                  ("PSO", PSO!, fitness, PSO_PARAMS),
                  ("NSGA2", NSGA2!, evaluate_multiobjective, NSGA2_PARAMS)],
    make_gif::Bool = true
    )

    println("\nRunning behaviour visualizations...")

    for dataset in datasets
        landscape  = load_landscape(dataset)
        n_bits     = landscape.n_features
        local_optima = get_local_optima(landscape; k=1)  # k=1 for speed

        dataset_short = split(dataset, ".")[1]
        out_path      = joinpath(@__DIR__, "..", "img", dataset_short, "behavior")
        mkpath(out_path)

        for (alg_name, alg_fn, eval_fn, params) in algorithms
            println("  $alg_name on $dataset_short...")

            history, _, _ = alg_fn(landscape, POPSIZE, GENERATIONS, eval_fn; params...)

            # Static panel figure
            fig = plot_behavior_panel(landscape, history, alg_name, local_optima)
            save(joinpath(out_path, "$(dataset_short)_$(alg_name)_behavior.png"), fig)

            # GIF animation (skip if no row-6 path tracking, i.e. NSGA2)
            if make_gif && size(history, 1) >= 6
                gif_path = joinpath(out_path, "$(dataset_short)_$(alg_name)_behavior.gif")
                animate_behavior(landscape, history, alg_name, gif_path;
                    framerate = 20, skip = max(1, GENERATIONS ÷ 150))
            end

            println("    Saved to $out_path")
        end
    end
end

# ============== Test Behavior ===============

function test_behavior()

    dataset = first(keys(DATASETS))
    landscape = load_landscape(dataset)

    history, _, _ = GA!(landscape, 10, 200, fitness)
    best_path = Int.(history[6, :])

    n_bits = landscape.n_features
    local_optima = get_local_optima(landscape; k=NEIGHBORHOOD_SIZE)

    f1 = plot_landscape_with_path(landscape, best_path, local_optima)
    f2 = plot_landscape_polar_with_path(landscape, best_path)
    f3 = plot_hinged_map_with_path(landscape, best_path, local_optima)

    g, opt_index_map, basin_map = build_LON(landscape.fitnesses, local_optima, n_bits; k=NEIGHBORHOOD_SIZE)
    basin_sizes = compute_basin_sizes(basin_map, local_optima)
    f4 = plot_lon_with_path(g, landscape, opt_index_map, basin_map, basin_sizes, best_path)

    dataset_short = split(dataset, ".")[1]
    out_path = joinpath(@__DIR__, "..", "img_behavior_test")
    mkpath(out_path)

    save(joinpath(out_path, "$(dataset_short)_landscape.png"), f1)
    save(joinpath(out_path, "$(dataset_short)_landscape_polar.png"), f2)
    save(joinpath(out_path, "$(dataset_short)_hinged_bitstring_map.png"), f3)
    save(joinpath(out_path, "$(dataset_short)_lon.png"), f4)

end

main()