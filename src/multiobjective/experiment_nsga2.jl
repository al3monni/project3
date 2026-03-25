include("nsga2.jl")
using Statistics
using Printf

# =========================================================
# UTILIDADES
# =========================================================

function unique_pareto_front(pareto_front::Vector{Individual})
    unique_dict = Dict{String, Individual}()

    for ind in pareto_front
        key = join(Int.(ind.bits))
        if !haskey(unique_dict, key)
            unique_dict[key] = ind
        end
    end

    return collect(values(unique_dict))
end

function summarize_pareto_front(pareto_front::Vector{Individual})
    unique_front = unique_pareto_front(pareto_front)

    best_acc = maximum(ind.objectives[1] for ind in unique_front)
    min_features = minimum(Int(-ind.objectives[2]) for ind in unique_front)
    pareto_size = length(unique_front)

    return best_acc, min_features, pareto_size, unique_front
end

function mean_std(v::Vector{<:Real})
    if length(v) == 1
        return Float64(v[1]), 0.0
    end
    return mean(v), std(v)
end

function write_results_csv(filename::String, rows::Vector{NamedTuple})
    open(filename, "w") do io
        println(io, "landscape,run,seed,best_accuracy,min_features,pareto_size")

        for row in rows
            println(
                io,
                string(
                    row.landscape, ",",
                    row.run, ",",
                    row.seed, ",",
                    row.best_accuracy, ",",
                    row.min_features, ",",
                    row.pareto_size
                )
            )
        end
    end
end

# =========================================================
# EXPERIMENTO SOBRE UN LANDSCAPE
# =========================================================

function run_experiment_on_landscape(filepath::String;
                                     n_runs::Int = 10,
                                     popsize::Int = 100,
                                     generations::Int = 50,
                                     pc::Float64 = 0.9,
                                     base_seed::Int = 100)

    landscape = load_landscape(filepath)

    println("\n======================================================")
    println("Landscape: ", landscape.name)
    println("n_features: ", landscape.n_features)
    println("n_points: ", length(landscape.mean_accuracies))
    println("n_runs: ", n_runs)
    println("popsize: ", popsize)
    println("generations: ", generations)
    println("pc: ", pc)
    println("pm: ", 1.0 / landscape.n_features)
    println("======================================================")

    best_accuracies = Float64[]
    min_features_list = Int[]
    pareto_sizes = Int[]

    all_rows = NamedTuple[]

    best_run_front = Vector{Individual}()
    best_run_acc = -Inf
    best_run_id = 0

    for run in 1:n_runs
        seed = base_seed + run - 1

        println("\n---------- Run $run / $n_runs | seed = $seed ----------")

        _, pareto_front = nsga2(
            landscape;
            popsize = popsize,
            generations = generations,
            pc = pc,
            pm = 1.0 / landscape.n_features,
            seed = seed
        )

        best_acc, min_feats, pareto_size, unique_front = summarize_pareto_front(pareto_front)

        push!(best_accuracies, best_acc)
        push!(min_features_list, min_feats)
        push!(pareto_sizes, pareto_size)

        push!(all_rows, (
            landscape = filepath,
            run = run,
            seed = seed,
            best_accuracy = best_acc,
            min_features = min_feats,
            pareto_size = pareto_size
        ))

        @printf("Run %d resumen | best_acc = %.6f | min_features = %d | pareto_size = %d\n",
                run, best_acc, min_feats, pareto_size)

        if best_acc > best_run_acc
            best_run_acc = best_acc
            best_run_front = unique_front
            best_run_id = run
        end
    end

    mean_acc, std_acc = mean_std(best_accuracies)
    mean_feat, std_feat = mean_std(Float64.(min_features_list))
    mean_psize, std_psize = mean_std(Float64.(pareto_sizes))

    println("\n================ RESUMEN FINAL ================")
    println("Landscape: ", filepath)
    @printf("Best accuracy medio      = %.6f ± %.6f\n", mean_acc, std_acc)
    @printf("Min features medio       = %.3f ± %.3f\n", mean_feat, std_feat)
    @printf("Pareto size medio        = %.3f ± %.3f\n", mean_psize, std_psize)
    println("Mejor run por accuracy   = Run ", best_run_id)
    println("================================================")

    return (
        landscape = landscape,
        rows = all_rows,
        best_accuracies = best_accuracies,
        min_features_list = min_features_list,
        pareto_sizes = pareto_sizes,
        best_run_front = best_run_front,
        best_run_id = best_run_id,
        mean_acc = mean_acc,
        std_acc = std_acc,
        mean_feat = mean_feat,
        std_feat = std_feat,
        mean_psize = mean_psize,
        std_psize = std_psize
    )
end

# =========================================================
# EXPERIMENTO GLOBAL
# =========================================================

files = [
    "01-breast-w_lr_F.h5",
    "05-credit-a_rf_F.h5",
    "08-letter-r_knn_F.h5"
]

n_runs = 10
popsize = 100
generations = 50
pc = 0.9
base_seed = 100

all_rows = NamedTuple[]
all_summaries = []

for file in files
    result = run_experiment_on_landscape(
        file;
        n_runs = n_runs,
        popsize = popsize,
        generations = generations,
        pc = pc,
        base_seed = base_seed
    )

    append!(all_rows, result.rows)
    push!(all_summaries, result)
end

# =========================================================
# GUARDAR CSV
# =========================================================

csv_filename = "nsga2_results.csv"
write_results_csv(csv_filename, all_rows)

println("\nCSV guardado en: ", csv_filename)

# =========================================================
# RESUMEN GLOBAL FINAL
# =========================================================

println("\n================ RESUMEN GLOBAL NSGA-II ================")
for res in all_summaries
    println("\nLandscape: ", res.landscape.name)
    @printf("Best accuracy medio = %.6f ± %.6f\n", res.mean_acc, res.std_acc)
    @printf("Min features medio  = %.3f ± %.3f\n", res.mean_feat, res.std_feat)
    @printf("Pareto size medio   = %.3f ± %.3f\n", res.mean_psize, res.std_psize)
    println("Mejor run ID        = ", res.best_run_id)
end
println("========================================================")