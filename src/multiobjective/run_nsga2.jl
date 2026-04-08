include("nsga2.jl")

landscape = load_landscape("01-breast-w_lr_F.h5")

println("===================================")
println("Landscape: ", landscape.name)
println("Número de features: ", landscape.n_features)
println("Número de puntos: ", length(landscape.mean_accuracies))
println("===================================")

population, pareto_front = nsga2(
    landscape;
    popsize = 100,
    generations = 50,
    pc = 0.9,
    pm = 1.0 / landscape.n_features,
    seed = 42
)

print_pareto_front(pareto_front, max_show = 20)