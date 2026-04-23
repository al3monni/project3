using Graphs
using GraphPlot
using CairoMakie
using GraphMakie
using NetworkLayout
using CSV, DataFrames

include("utils.jl")

# =========================================================
# FITNESS LANDSCAPE VISUALIZATION
# =========================================================

# =========================================================
# 1. Flat Fitness Landscape Visualization
# =========================================================

function plot_landscape(fitness_lookup::Vector{Float32})

    # extract the individuals
    x = eachindex(fitness_lookup)

    f = Figure(size = (900, 500))
    
    ax = Axis(
        f[1, 1],
        title = "Fitness Landscape",
        xlabel = "Individual index",
        ylabel = "Fitness"
    )

    lines!(ax, x, fitness_lookup)
    scatter!(ax, x, fitness_lookup)

    return f
end

# =========================================================
# 2. Polar Fitness Landscape Visualization
# =========================================================

function plot_landscape_polar(
    fitness_lookup::Vector{Float32};
    base_radius::Real = 0.1,
    radial_scale::Real = 0.4,
    markersize::Real = 6,
    show_points::Bool = true,
    show_line::Bool = true
    )

    x, y = polar_coordinates(fitness_lookup; base_radius=base_radius, radial_scale=radial_scale)

    f = Figure(size = (800, 800))
    ax = Axis(
        f[1, 1],
        title = "Polar Fitness Landscape",
        aspect = DataAspect()
    )

    if show_line
        lines!(ax, [x; x[1]], [y; y[1]])
    end

    if show_points
        scatter!(ax, x, y, markersize = markersize)
    end

    hidedecorations!(ax)
    hidespines!(ax)

    return f
end

# =========================================================
# 3. Hinged Bitstring Map Visualization
# =========================================================

function hinged_bitstring_map(landscape::Vector{Float32}, local_optima::Vector{Int} = Int[])
    n = length(landscape)
    bits = ceil(Int, log2(n))

    bits += isodd(bits)  # make even
    half = bits ÷ 2

    x = Int[]
    y = Int[]

    for i in 1:n
        b = lpad(string(i, base=2), bits, '0')

        xi = parse(Int, b[1:half]; base=2)
        yi = parse(Int, b[half+1:end]; base=2)

        push!(x, xi)
        push!(y, yi)
    end

    f = Figure(size = (1200, 900))
    ax = Axis(f[1, 1],
        title = "Hinged Bitstring Map",
        xlabel = "First half",
        ylabel = "Second half"
    )

    # base plot
    map = heatmap!(ax, x, y, landscape, colormap = :viridis)

    # extract coordinates of local optima
    x_opt = x[local_optima]
    y_opt = y[local_optima]

    # overlay: highlight them
    scatter!(ax, x_opt, y_opt,
        color = :white,
        strokecolor = :black,
        strokewidth = 1
    )
    Colorbar(f[1, 2], map, label = "Fitness")
    return f
end

# =========================================================
# 4. Local Optima Network Visualization
# =========================================================

function hill_climb(start::Int, landscape::Vector{Float32}, n_bits::Int; k::Int=1)
    current = start
    while true
        neigh = neighbors(current, n_bits; k=k)
        # pick the neighbor with maximum fitness (greedy)
        best_fitness = landscape[current]
        best_neighbor = current
        for n in neigh
            if landscape[n] > best_fitness
                best_fitness = landscape[n]
                best_neighbor = n
            end
        end
        # stop if no improvement
        if best_neighbor == current
            return current  # local optimum reached
        end
        current = best_neighbor
    end
end # Debug OK 

function compute_basins(landscape::Vector{Float32}, local_optima::Vector{Int}, n_bits::Int; k::Int=1)
    n = length(landscape)
    basin_map = Dict{Int, Int}()  # point index → optimum index

    for i in 1:n
        basin_map[i] = hill_climb(i, landscape, n_bits; k=k)
    end

    return basin_map
end # Debug OK

function compute_basin_sizes(basin_map::Dict{Int,Int}, local_optima::Vector{Int})

    basin_sizes = Dict{Int, Int}()

    for opt in local_optima
        basin_sizes[opt] = 0
    end

    for (_, opt) in basin_map
        basin_sizes[opt] += 1
    end

    return basin_sizes
end # Debug OK

function build_LON(landscape::Vector{Float32}, local_optima::Vector{Int}, n_bits::Int; k::Int=1)
    
    # compute the number of local optima
    n_opt = length(local_optima)

    # create a graph with n nodes representing the local optima
    g = DiGraph(n_opt)

    # map the local optima to their graph indices
    opt_index_map = Dict(opt => idx for (idx, opt) in enumerate(local_optima))

    # compute basin for each point
    basin_map = compute_basins(landscape, local_optima, n_bits; k=k)

    # create an edge set to avoid duplicates
    edges_added = Set{Tuple{Int,Int}}()

    for i in 1:length(landscape)
        current_basin = basin_map[i]
        neigh = neighbors(i, n_bits; k=k)
        for n in neigh
            neighbor_basin = basin_map[n]
            if neighbor_basin != current_basin
                # add directed edge from current basin to neighbor basin
                src = opt_index_map[current_basin]
                dst = opt_index_map[neighbor_basin]
                if (src,dst) ∉ edges_added
                    add_edge!(g, src, dst)
                    push!(edges_added, (src,dst))
                end
            end
        end
    end

    return g, opt_index_map, basin_map
end # Debug OK

function export_LON(
    landscape::Vector{Float32},
    g::DiGraph,
    opt_index_map::Dict{Int,Int},
    basin_sizes::Dict{Int,Int}
    )

    # Build reverse map (graph idx → optimum)
    index_to_opt = Dict(v => k for (k,v) in opt_index_map)

    # Nodes Table
    nodes_df = DataFrame(
        Id = Int[],
        Label = String[],
        Fitness = Float32[],
        BasinSize = Int[]
    )

    for i in 1:nv(g)
        opt = index_to_opt[i]

        push!(nodes_df, (
            i,
            string(opt),
            landscape[opt],
            basin_sizes[opt]
        ))
    end

    # Edges Table
    edges_df = DataFrame(
        Source = Int[],
        Target = Int[],
    )

    for e in edges(g)
        s = src(e)
        d = dst(e)
        push!(edges_df, (s, d))
    end

    out_path = mkpath(joinpath(@__DIR__, "..", "csv"))

    CSV.write(joinpath(out_path, "nodes.csv"), nodes_df)
    CSV.write(joinpath(out_path, "edges.csv"), edges_df)

    # println("Export complete: nodes.csv and edges.csv saved in $out_path.")
end # Debug OK

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

    positions = layout(g)

    # --- 5. Plot ---
    f = Figure(size = figsize)
    ax = Axis(f[1, 1])

    graphplot!(ax, g;
        layout = positions,

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

    return f, positions
end

# =========================================================
# EXPERIMENTAL VISUALIZATIONS
# =========================================================

function plot_evolution(history, landscape, algorithm)

    if algorithm == "GA" || algorithm == "PSO"
        min = minimum(landscape.fitnesses)
        max = maximum(landscape.fitnesses)
        y_label = "Fitness"
    else
        min = minimum(landscape.accuracies)
        max = maximum(landscape.accuracies)
        y_label = "Accuracy"
    end

    sigma = (max - min) * 0.1  # add 10% margin above max for better visualization

    generations = 1:size(history, 2)
    f = Figure(size = (900, 500))
    ax = Axis(
        f[1, 1],
        title = "$algorithm on $(split(landscape.name, ".")[1])",
        xlabel = "Generation",
        ylabel = y_label,
        limits = (0, GENERATIONS, min, max + sigma)
    )

    # Plot mean fitness
    lines!(ax, generations, history[3, :], label = "Mean Fitness", color=:blue)

    # Plot min/max as a shaded area
    band!(ax, generations, history[1, :], history[2, :], color=:blue, alpha=0.3, label="Min-Max Range")

    # Plot optimal fitness as a dashed line
    hlines!(ax, [max], color=:red, linestyle=:dash, label="Optimal $y_label")

    axislegend(position = :rb)
    return f
end

function plot_entropy(history, title)
    generations = 1:size(history, 2)
    f = Figure(size = (900, 500))
    ax = Axis(
        f[1, 1],
        title = title,
        xlabel = "Generation",
        ylabel = "Entropy"
    )

    lines!(ax, generations, history[5, :], label = "Population Entropy", color=:red)
    axislegend(ax)    
    return f
end

function plot_pareto_front(pareto_front, title)
    f = Figure(size = (900, 500))
    ax = Axis(
        f[1, 1],
        title = title,
        xlabel = "Accuracy",
        ylabel = "Negative Number of Features"
    )

    scatter!(ax, [p.objectives[1] for p in pareto_front], [p.objectives[2] for p in pareto_front], color=:green, label="Pareto Front")
    axislegend(ax)    
    return f
end