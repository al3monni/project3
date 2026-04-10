include("data_load.jl")
include("utils.jl")
include("visualization.jl")
include("local_optima_network.jl")
include("plot_lon.jl")

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

    for i in eachindex(datasets)

        # 0.1 Load the landscape and get the local optima
        landscape = load_landscape(datasets[i])
        n = length(landscape)
        n_bits = ceil(Int, log2(n))
        local_optima = get_local_optima(landscape)

        # ============== Visualizations ==============

        f1 = plot_landscape(landscape)
        f2 = plot_landscape_polar(landscape)
        #println("Local optima: $local_optima")
        f3 = hinged_bitstring_map(landscape, local_optima)

        # ==================== LON ====================
        
        # Build LON
        g, opt_index_map, basin_map = build_LON(landscape, local_optima, n_bits)

        # Compute basin sizes
        basin_sizes = compute_basin_sizes(basin_map, local_optima)

        # Export LON
        # export_LON(landscape, g, opt_index_map, basin_sizes)

        # Plot LON
        f4 = plot_lon(g, landscape, opt_index_map, basin_sizes)

        display(f1)
        display(f2)
        display(f3)
        display(f4)

        out_path = mkpath(joinpath(@__DIR__, "..", "img"))

        save("$out_path/$(datasets[i])_landscape.png",              f1)
        save("$out_path/$(datasets[i])_landscape_polar.png",        f2)
        save("$out_path/$(datasets[i])_hinged_bitstring_map.png",   f3)
        save("$out_path/$(datasets[i])_lon.png",                    f4)

    end
end

main()