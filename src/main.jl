include("data_load.jl")
include("utils.jl")
include("visualization.jl")

# ============= Parameters =============

const dataset_name = "accuracies"
const datasets = [
    "01-breast-w_lr_F.h5",
    "05-credit-a_rf_F.h5",
    "08-letter-r_knn_F.h5"
]
const penalty = 0.01
const n = 16
const m = 1.0f0
const s = 4

# ======================================

landscape = load_landscape(datasets[1])

local_optima = get_local_optima(landscape)
println("Local optima: $local_optima")

hinged_bitstring_map(landscape, local_optima)