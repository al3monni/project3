include("data_load.jl")
include("utils.jl")
include("visualization.jl")
include("local_optima_network.jl")

# ============== Global Parameters =============

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

# ==================== Main ====================

function main()

    # 0.1 Load the landscape and get the local optima
    landscape = load_landscape(datasets[1])
    local_optima = get_local_optima(landscape)

    # 0.2 Get the number of bits needed to represent the landscape
    n = length(landscape)
    n_bits = ceil(Int, log2(n))

    #println("Local optima: $local_optima")
    #hinged_bitstring_map(landscape, local_optima)

    # =============================================================

    # Build LON
    g, opt_index_map, basin_map = build_LON(landscape, local_optima, n_bits)

    # Compute basin sizes
    basin_sizes = compute_basin_sizes(basin_map, local_optima)

    # Export LON
    export_LON(landscape, g, opt_index_map, basin_sizes)

    # Plot LON

end

main()