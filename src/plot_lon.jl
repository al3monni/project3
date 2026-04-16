using Graphs
using CairoMakie
using GraphMakie
using NetworkLayout

function plot_lon(
    g::DiGraph,
    landscape,
    opt_index_map,
    basin_sizes;
    layout_algo = :spring,
    node_scale = 40,
    log_sizes = true,
    show_arrows = true,
    edge_alpha = 0.2,
    figsize = (900, 800)
    )

    N = nv(g)

    # Create inverse mapping: node index to optimum
    idx_to_opt = Dict(idx => opt for (opt, idx) in opt_index_map)

    # --- 1. Node data ---
    node_sizes = [basin_sizes[idx_to_opt[i]] for i in 1:N]
    fitness = [landscape[idx_to_opt[i]] for i in 1:N]

    # --- 2. Normalize sizes ---
    if log_sizes
        sizes = log.(node_sizes .+ 1)
    else
        sizes = node_sizes
    end

    sizes_norm = 10 .+ node_scale .* (sizes .- minimum(sizes)) ./
                        (maximum(sizes) - minimum(sizes) + eps())

    # --- 3. Normalize fitness for color ---
    fitness_norm = (fitness .- minimum(fitness)) ./
                   (maximum(fitness) - minimum(fitness) + eps())

    # --- 4. Layout নির্বাচন ---
    layout = if layout_algo == :spring
        Spring(; iterations=500)
    elseif layout_algo == :kamada
        KamadaKawai()
    elseif layout_algo == :spectral
        Spectral()
    else
        Spring()
    end

    # --- 5. Plot ---
    f = Figure(size = figsize)
    ax = Axis(f[1, 1])

    graphplot!(ax, g;
        layout = layout,

        # Nodes
        node_size = sizes_norm,
        node_color = fitness_norm,
        colormap = :viridis,

        # Edges
        edge_color = (:black, edge_alpha),
        edge_width = 1.0,

        # Directed
        arrow_show = show_arrows,
        arrow_size = 8
    )

    # Colorbar
    Colorbar(f[1, 2], colormap = :viridis, label = "Fitness")

    return f
end