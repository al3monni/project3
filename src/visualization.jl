using Graphs
using GraphPlot
using CairoMakie
using GraphMakie
using NetworkLayout
using CSV, DataFrames

# =========================================================
# FITNESS LANDSCAPE VISUALIZATION
# =========================================================

# =========================================================
# 1. Flat Fitness Landscape Visualization
# =========================================================

# max_points: thin the line for large landscapes (still marks all local optima exactly).
# Set max_points=0 to disable thinning.
function plot_landscape(fitness_lookup::Vector{Float32}, local_optima::Vector{Int} = Int[];
                         show_points::Bool = false, max_points::Int = 4000)
    n = length(fitness_lookup)
    if max_points > 0 && n > max_points
        step = ceil(Int, n / max_points)
        idx  = collect(1:step:n)
        x    = idx
        y    = fitness_lookup[idx]
    else
        x = collect(eachindex(fitness_lookup))
        y = fitness_lookup
    end

    f  = Figure(size = (900, 500))
    ax = Axis(f[1, 1],
        title  = "Fitness Landscape  (n=$(n) points$(n > max_points > 0 ? ", thinned to $(length(x))" : ""))",
        xlabel = "Individual index",
        ylabel = "Fitness"
    )

    lines!(ax, x, y)
    if show_points
        scatter!(ax, x, y, markersize = 4)
    end
    if !isempty(local_optima)
        scatter!(ax, local_optima, fitness_lookup[local_optima],
            color = :red, markersize = 8, label = "Local optima ($(length(local_optima)))")
        axislegend(ax, position = :rb)
    end
    return f
end

# Sorted fitness landscape: x = rank (best→worst), y = fitness.
# Complements the index-based view for large, dense landscapes.
function plot_landscape_sorted(fitness_lookup::Vector{Float32}, local_optima::Vector{Int} = Int[])
    sorted_fit = sort(fitness_lookup, rev = true)
    n = length(sorted_fit)

    f  = Figure(size = (900, 500))
    ax = Axis(f[1, 1],
        title  = "Sorted Fitness Landscape  (n=$(n))",
        xlabel = "Rank (best → worst)",
        ylabel = "Fitness"
    )
    lines!(ax, 1:n, sorted_fit, color = :steelblue)

    if !isempty(local_optima)
        opt_fits = sort(fitness_lookup[local_optima], rev = true)
        scatter!(ax, 1:length(opt_fits), opt_fits,
            color = :red, markersize = 6, label = "Local optima ($(length(local_optima)))")
        axislegend(ax, position = :rt)
    end
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

    axislegend(ax, position = :rb)
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
# =========================================================
# SYNTHETIC LANDSCAPE VISUALIZATION
# =========================================================

# The triangle function is phenotypic: f depends only on ‖b‖ (number of
# active bits), not on which bits are set. The right visualization is
# therefore the 1D phenotypic landscape: x = ‖b‖, y = f(‖b‖).
# Basin sizes are C(n, k) — the binomial coefficient — shown as a bar
# overlay to convey how many bitstrings share each fitness level.

function plot_triangle_phenotype(
    filename::String;
    show_basins::Bool = true,
    figsize::Tuple = (900, 500)
    )

    n = CONFIG["datasets"][filename]["n"]
    m = CONFIG["datasets"][filename]["m"]
    s = CONFIG["datasets"][filename]["s"]

    if filename == "triangle"
        # triangle_phenotype(k, m, s) takes norm_b directly
        f_val = [triangle_phenotype(k, m, s) for k in 0:n]
    elseif filename == "asymmetric"
        f_val = [Int(TRIANGLE_ASYMMETRIC_TABLE[k + 1]) for k in 0:n]
    else
        error("Not a synthetic landscape: $filename")
    end

    xs          = 0:n
    peak_val    = maximum(f_val)
    peaks       = [k for k in 0:n if f_val[k + 1] == peak_val]
    basin_sizes = [binomial(n, k) for k in 0:n]

    fig = Figure(size = figsize)

    if show_basins
        ax1 = Axis(fig[1, 1],
            title             = "Triangle Landscape  (n=$n, m=$m, s=$s)",
            xlabel            = "Number of active bits ‖b‖",
            ylabel            = "Fitness f(‖b‖)",
            yticklabelcolor   = :blue,
            ylabelcolor       = :blue
        )
        ax2 = Axis(fig[1, 1],
            ylabel            = "Basin size  C(n, ‖b‖)",
            yaxisposition     = :right,
            yticklabelcolor   = :gray60,
            ylabelcolor       = :gray60,
            ygridvisible      = false,
            xgridvisible      = false,
            backgroundcolor   = :transparent
        )
        hidespines!(ax2)
        hidexdecorations!(ax2)
        linkxaxes!(ax1, ax2)

        barplot!(ax2, collect(xs), Float64.(basin_sizes),
            color = (:gray80, 0.5), strokewidth = 0)
        lines!(ax1, collect(xs), Float64.(f_val), color = :blue, linewidth = 2)
        scatter!(ax1, collect(xs), Float64.(f_val), color = :blue, markersize = 6)
        scatter!(ax1, peaks, Float64.(f_val[peaks .+ 1]),
            color = :red, markersize = 14, marker = :star5,
            label = "Local optima ($(sum(basin_sizes[p+1] for p in peaks)) bitstrings)")
        axislegend(ax1, position = :lt)
    else
        ax1 = Axis(fig[1, 1],
            title  = "Triangle Landscape  (n=$n, m=$m, s=$s)",
            xlabel = "Number of active bits ‖b‖",
            ylabel = "Fitness f(‖b‖)"
        )
        lines!(ax1, collect(xs), Float64.(f_val), color = :blue, linewidth = 2)
        scatter!(ax1, collect(xs), Float64.(f_val), color = :blue, markersize = 6)
        scatter!(ax1, peaks, Float64.(f_val[peaks .+ 1]),
            color = :red, markersize = 14, marker = :star5, label = "Local optima")
        axislegend(ax1, position = :lt)
    end

    return fig
end

# =========================================================
# ALGORITHM BEHAVIOUR VISUALIZATIONS
# =========================================================

# Build a proper 2D matrix for heatmap from a flat fitness vector.
# Returns (xs_grid, ys_grid, Z) where xs_grid = 0:nx-1, ys_grid = 0:ny-1,
# Z is an nx×ny Float32 matrix — NaN where no bitstring maps (only cell 0,0).
function _hinged_matrix(fitnesses::Vector{Float32}, n_bits::Int)

    # NO more bits_even
    half = n_bits ÷ 2

    nx = 2^half
    ny = 2^(n_bits - half)

    Z = fill(NaN32, nx, ny)

    for i in eachindex(fitnesses)
        xi, yi = _hinged_coords(i, n_bits)

        # fill only valid cells
        Z[Int(xi) + 1, Int(yi) + 1] = fitnesses[i]
    end

    return 0:nx-1, 0:ny-1, Z
end

# Helper: convert a flat bitstring index to (x, y) in the hinged grid
function _hinged_coords(idx::Int, n_bits::Int)
    half = n_bits ÷ 2

    # IMPORTANT: use EXACT number of bits (no artificial padding)
    b = lpad(string(idx, base = 2), n_bits, '0')

    x = parse(Int, b[1:half]; base = 2)
    y = parse(Int, b[half+1:end]; base = 2)

    return Float64(x), Float64(y)
end

# ── Static multi-panel behaviour figure ──────────────────────────────────────
#
# Panel layout (for algorithms that track best-individual path, i.e. GA/PSO):
#   [1,1]  Fitness evolution  (best / mean / min band)
#   [1,2]  Population entropy over generations
#   [2, 1:2]  Fading trail of best individual on hinged bitstring map
#              (recent generations = opaque red, older = faint)
#
# For NSGA2 (history has 5 rows, no best-index), only the top two panels
# are shown.

# Behaviour PNG: just the best-individual trail on the hinged bitstring map.
# For NSGA2 (no row-6 best index) shows the final population spread only.
# Legend is placed outside the axis to the right.
function plot_behavior_panel(
    landscape::Landscape,
    history::Matrix{Float64},
    snapshots::Dict{Int, Vector{Int}},
    algorithm_name::String,
    local_optima::Vector{Int} = Int[];
    figsize::Tuple = (1100, 700)
    )

    n_gens   = size(history, 2)
    has_path = size(history, 1) >= 6
    has_snap = !isempty(snapshots)
    n_bits   = landscape.n_features
    dataset_name = split(landscape.name, ".")[1]

    # Proper 2D matrix for heatmap → full grid, no Makie guesswork
    xs_grid, ys_grid, Z = _hinged_matrix(landscape.fitnesses, n_bits)

    fig = Figure(size = figsize)

    map_title = has_path ?
        "$algorithm_name on $dataset_name — best-individual trail  (bright red = recent)" :
        "$algorithm_name on $dataset_name — final population spread"

    ax_map = Axis(fig[1, 1],
        title  = map_title,
        xlabel = "First-half bits",
        ylabel = "Second-half bits"
    )

    hm = heatmap!(ax_map, xs_grid, ys_grid, Z, colormap = :viridis, nan_color = :black)
    Colorbar(fig[1, 2], hm, label = "Fitness")

    # Local optima markers
    if !isempty(local_optima)
        ox = [_hinged_coords(o, n_bits)[1] for o in local_optima]
        oy = [_hinged_coords(o, n_bits)[2] for o in local_optima]
        scatter!(ax_map, ox, oy,
            color = (:white, 0.7), strokecolor = :black, strokewidth = 0.8,
            markersize = 7, label = "Local optima")
    end

    # Final population scatter
    if has_snap
        last_snap = snapshots[maximum(keys(snapshots))]
        px = Float64[_hinged_coords(i, n_bits)[1] for i in last_snap]
        py = Float64[_hinged_coords(i, n_bits)[2] for i in last_snap]
        scatter!(ax_map, px, py,
            color = (:white, 0.55), markersize = 6, strokewidth = 0,
            label = "Population (last gen)")
    end

    # Fading best-individual trail (GA / PSO only)
    if has_path
        best_path = Int.(history[6, :])
        xs = Float64[_hinged_coords(idx, n_bits)[1] for idx in best_path]
        ys = Float64[_hinged_coords(idx, n_bits)[2] for idx in best_path]

        alphas = range(0.08, 1.0, length = n_gens)
        for i in 1:n_gens
            scatter!(ax_map, [xs[i]], [ys[i]],
                color = RGBAf(1.0, 0.15, 0.1, Float32(alphas[i])),
                markersize = 9)
        end
        # Dummy entry for the legend (single representative point)
        scatter!(ax_map, [xs[end]], [ys[end]],
            color = RGBAf(1.0, 0.15, 0.1, 1.0), markersize = 9,
            label = "Best individual trail (fades with age)")

        # Red star = current (final) best
        scatter!(ax_map, [xs[end]], [ys[end]],
            color = :red, markersize = 18, marker = :star5,
            label = "Best individual (final gen)")
    end

    # Legend outside the axis, to the right of the colorbar
    Legend(fig[1, 3], ax_map, framevisible = true)

    return fig
end

# ── GIF animation of algorithm behaviour ─────────────────────────────────────
#
# Produces an animated GIF (requires CairoMakie backend + FFMPEG or Rsvg).
# Left panel: fitness/entropy curves drawn progressively.
# Right panel: best individual's position on the hinged map with a fading trail.
#
# skip: render every `skip`-th generation to keep file size manageable.

# function animate_behavior(
#     landscape::Landscape,
#     history::Matrix{Float64},
#     snapshots::Dict{Int, Vector{Int}},
#     algorithm_name::String,
#     out_path::String;
#     framerate::Int = 15,
#     skip::Int = 1
#     )

#     has_path = size(history, 1) >= 6     # GA / PSO: row 6 = best individual index
#     has_snap = !isempty(snapshots)
#     n_gens   = size(history, 2)
#     n_bits   = landscape.n_features
#     dataset_name = split(landscape.name, ".")[1]

#     frames = collect(1:skip:n_gens)

#     # ── Precompute everything that does not change frame-to-frame ─────────
#     xs_grid, ys_grid, Z = _hinged_matrix(landscape.fitnesses, n_bits)

#     if has_path
#         best_path = Int.(history[6, :])
#         trail_x   = [_hinged_coords(best_path[g], n_bits)[1] for g in 1:n_gens]
#         trail_y   = [_hinged_coords(best_path[g], n_bits)[2] for g in 1:n_gens]
#     end

#     snap_gens = has_snap ? sort(collect(keys(snapshots))) : Int[]

#     # ── Axis limits ───────────────────────────────────────────────────────
#     f_min  = minimum(history[1, :])
#     f_max  = maximum(history[2, :])
#     f_sig  = max((f_max - f_min) * 0.1, 1e-6)
#     e_max  = maximum(history[5, :])
#     e_sig  = e_max * 0.1

#     half = n_bits ÷ 2
#     map_xlims = (-0.5, Float64(2^half - 1) + 0.5)
#     map_ylims = (-0.5, Float64(2^(n_bits - half) - 1) + 0.5)

#     # ── Figure ────────────────────────────────────────────────────────────
#     fig = Figure(size = (1100, 480))

#     # Left: fitness curve (blue) + entropy overlay (red, right axis)
#     ax_fit = Axis(fig[1, 1],
#         title  = "$algorithm_name on $dataset_name",
#         xlabel = "Generation",
#         ylabel = "Fitness",
#         limits = (1, n_gens, f_min - f_sig, f_max + f_sig)
#     )
#     ax_ent = Axis(fig[1, 1],
#         ylabel          = "Entropy (bits)",
#         yaxisposition   = :right,
#         yticklabelcolor = :crimson,
#         ylabelcolor     = :crimson,
#         ygridvisible    = false,
#         xgridvisible    = false,
#         backgroundcolor = :transparent,
#         limits          = (1, n_gens, 0, e_max + e_sig)
#     )
#     hidespines!(ax_ent); hidexdecorations!(ax_ent)
#     linkxaxes!(ax_fit, ax_ent)

#     gen_obs  = Observable(Int[])
#     fit_obs  = Observable(Float64[])
#     best_obs = Observable(Float64[])
#     ent_obs  = Observable(Float64[])

#     lines!(ax_fit, gen_obs, fit_obs,  color = :steelblue, linewidth = 2, label = "Mean fitness")
#     lines!(ax_fit, gen_obs, best_obs, color = :darkblue,  linewidth = 1.5, linestyle = :dash, label = "Best fitness")
#     lines!(ax_ent, gen_obs, ent_obs,  color = :crimson,   linewidth = 2, label = "Entropy")
#     axislegend(ax_fit, position = :rb)

#     # Right: hinged map — shown for GA/PSO (trail + pop) and NSGA2 (pop only)
#     if has_path || has_snap
#         ax_map = Axis(fig[1, 2],
#             title  = has_path ? "Population & best-individual trajectory" : "Population spread",
#             xlabel = "First-half bits",
#             ylabel = "Second-half bits",
#             limits = (map_xlims..., map_ylims...)
#         )
#         heatmap!(ax_map, xs_grid, ys_grid, Z,
#             colormap = :viridis,
#             nan_color = (:white, 0.0)
#         )

#         # Population scatter observable
#         pop_x_obs = Observable(Float64[])
#         pop_y_obs = Observable(Float64[])
#         scatter!(ax_map, pop_x_obs, pop_y_obs,
#             color = (:white, 0.6), markersize = 6, strokewidth = 0,
#             label = "Population")

#         if has_path
#             # Fading trail observable (accumulates)
#             trail_x_obs = Observable(Float64[])
#             trail_y_obs = Observable(Float64[])
#             trail_c_obs = Observable(RGBAf[])
#             scatter!(ax_map, trail_x_obs, trail_y_obs,
#                 color = trail_c_obs, markersize = 6,
#                 label = "Best individual trail")

#             # Current best position (single point, updated each frame)
#             cur_x_obs = Observable([trail_x[1]])
#             cur_y_obs = Observable([trail_y[1]])
#             scatter!(ax_map, cur_x_obs, cur_y_obs,
#                 color = :red, markersize = 16, marker = :star5,
#                 label = "Current best individual")
#         end

#         axislegend(ax_map, position = :rt)
#     end

#     # ── Record frames ─────────────────────────────────────────────────────
#     record(fig, out_path, frames; framerate = framerate) do gen
#         push!(gen_obs[],  gen)
#         push!(fit_obs[],  history[3, gen])
#         push!(best_obs[], history[2, gen])
#         push!(ent_obs[],  history[5, gen])
#         notify(gen_obs); notify(fit_obs); notify(best_obs); notify(ent_obs)

#         if has_path || has_snap
#             # Population scatter: nearest snapshot
#             if has_snap
#                 si = searchsortedlast(snap_gens, gen)
#                 sg = si == 0 ? snap_gens[1] : snap_gens[si]
#                 pop_x_obs[] = Float64[_hinged_coords(i, n_bits)[1] for i in snapshots[sg]]
#                 pop_y_obs[] = Float64[_hinged_coords(i, n_bits)[2] for i in snapshots[sg]]
#             end

#             if has_path
#                 # Append trail point with alpha proportional to age
#                 alpha = 0.15 + 0.85 * (gen / n_gens)
#                 push!(trail_x_obs[], trail_x[gen])
#                 push!(trail_y_obs[], trail_y[gen])
#                 push!(trail_c_obs[], RGBAf(1.0, 0.3, 0.1, Float32(alpha)))
#                 notify(trail_x_obs); notify(trail_y_obs); notify(trail_c_obs)
#                 cur_x_obs[] = [trail_x[gen]]
#                 cur_y_obs[] = [trail_y[gen]]
#             end
#         end
#     end
# end

function animate_behavior(
    landscape::Landscape,
    history::Matrix{Float64},
    snapshots::Dict{Int, Vector{Int}},
    algorithm_name::String,
    out_path::String;
    framerate::Int = 15,
    skip::Int = 1
)
    has_path = size(history, 1) >= 6
    has_snap = !isempty(snapshots)

    n_gens = size(history, 2)
    n_bits = landscape.n_features
    dataset_name = split(landscape.name, ".")[1]

    frames = collect(1:skip:n_gens)

    xs_grid, ys_grid, Z = _hinged_matrix(landscape.fitnesses, n_bits)

    if has_path
        best_path = Int.(history[6, :])
        trail_x = [_hinged_coords(best_path[g], n_bits)[1] for g in 1:n_gens]
        trail_y = [_hinged_coords(best_path[g], n_bits)[2] for g in 1:n_gens]
    end

    snap_gens = has_snap ? sort(collect(keys(snapshots))) : Int[]

    half = n_bits ÷ 2
    map_xlims = (-0.5, Float64(2^half - 1) + 0.5)
    map_ylims = (-0.5, Float64(2^(n_bits - half) - 1) + 0.5)

    fig = Figure(size = (700, 650))

    ax_map = Axis(
        fig[1, 1],
        title = "$algorithm_name on $dataset_name — population and best trajectory",
        xlabel = "First-half bits",
        ylabel = "Second-half bits",
        limits = (map_xlims..., map_ylims...)
    )

    hm = heatmap!(
        ax_map,
        xs_grid,
        ys_grid,
        Z,
        colormap = :viridis,
        nan_color = (:white, 0.0)
    )

    Colorbar(fig[1, 2], hm, label = "Fitness")

    pop_x_obs = Observable(Float64[])
    pop_y_obs = Observable(Float64[])

    if has_snap
        scatter!(
            ax_map,
            pop_x_obs,
            pop_y_obs,
            color = (:white, 0.55),
            strokecolor = :black,
            strokewidth = 0.4,
            markersize = 6,
            label = "Population"
        )
    end

    if has_path
        trail_x_obs = Observable(Float64[])
        trail_y_obs = Observable(Float64[])
        trail_c_obs = Observable(RGBAf[])

        scatter!(
            ax_map,
            trail_x_obs,
            trail_y_obs,
            color = trail_c_obs,
            markersize = 7,
            label = "Best individual trajectory"
        )

        cur_x_obs = Observable([trail_x[1]])
        cur_y_obs = Observable([trail_y[1]])

        scatter!(
            ax_map,
            cur_x_obs,
            cur_y_obs,
            color = :red,
            markersize = 18,
            marker = :star5,
            label = "Current best individual"
        )
    end

    axislegend(ax_map, position = :rt)

    record(fig, out_path, frames; framerate = framerate) do gen

        if has_snap
            si = searchsortedlast(snap_gens, gen)
            sg = si == 0 ? snap_gens[1] : snap_gens[si]

            pop_x_obs[] = Float64[_hinged_coords(i, n_bits)[1] for i in snapshots[sg]]
            pop_y_obs[] = Float64[_hinged_coords(i, n_bits)[2] for i in snapshots[sg]]
        end

        if has_path
            alpha = 0.15 + 0.85 * (gen / n_gens)

            push!(trail_x_obs[], trail_x[gen])
            push!(trail_y_obs[], trail_y[gen])
            push!(trail_c_obs[], RGBAf(1.0, 0.3, 0.1, Float32(alpha)))

            notify(trail_x_obs)
            notify(trail_y_obs)
            notify(trail_c_obs)

            cur_x_obs[] = [trail_x[gen]]
            cur_y_obs[] = [trail_y[gen]]
        end
    end

    return fig
end

# =========================================================
# POPULATION SPREAD VISUALIZATIONS
# =========================================================

# ── Fitness density heatmap ───────────────────────────────────────────────────
#
# x = snapshot generation, y = fitness value (binned), color = number of
# individuals in that cell.  A horizontal band means the population is stuck;
# a rising band means it is converging toward high fitness.
# Works for all three algorithms (GA, PSO, NSGA2).

function plot_fitness_density(
    landscape::Landscape,
    snapshots::Dict{Int, Vector{Int}},
    algorithm_name::String;
    n_bins::Int  = 40,
    figsize::Tuple = (900, 500)
    )

    isempty(snapshots) && error("No snapshots to plot. Pass snapshot_every > 0 when running the algorithm.")

    gens      = sort(collect(keys(snapshots)))
    all_fits  = landscape.fitnesses
    fmin, fmax = minimum(all_fits), maximum(all_fits)
    edges     = range(fmin, fmax, length = n_bins + 1)

    # Build density matrix: rows = bins (y), cols = generations (x)
    density = zeros(Float64, n_bins, length(gens))
    best_per_gen = Float64[]

    for (ci, g) in enumerate(gens)
        fits = all_fits[snapshots[g]]
        push!(best_per_gen, maximum(fits))
        for fi in fits
            bin = min(searchsortedlast(edges, fi), n_bins)
            bin = max(bin, 1)
            density[bin, ci] += 1.0
        end
    end

    bin_centers = Float64[(edges[i] + edges[i+1]) / 2 for i in 1:n_bins]
    dataset_name = split(landscape.name, ".")[1]

    fig = Figure(size = figsize)
    ax  = Axis(fig[1, 1],
        title  = "$algorithm_name on $dataset_name — population fitness density",
        xlabel = "Generation",
        ylabel = "Fitness"
    )

    heatmap!(ax, Float64.(gens), bin_centers, density',
        colormap = :inferno)
    Colorbar(fig[1, 2], colormap = :inferno, label = "# individuals")

    # Overlay best individual's fitness per generation
    lines!(ax, Float64.(gens), best_per_gen,
        color = :cyan, linewidth = 2, label = "Best fitness")
    axislegend(ax, position = :rb)

    return fig
end

# ── Population snapshot grid ──────────────────────────────────────────────────
#
# Shows up to 9 evenly-spaced snapshots as a grid of hinged bitstring maps.
# Each panel: all individuals as scatter (color = fitness), local optima as
# white circles.  Useful for both report and presentation.

function plot_population_snapshots(
    landscape::Landscape,
    snapshots::Dict{Int, Vector{Int}},
    algorithm_name::String,
    local_optima::Vector{Int} = Int[];
    max_panels::Int  = 9,
    figsize::Tuple   = (1200, 1200)
    )

    isempty(snapshots) && error("No snapshots to plot.")

    gens = sort(collect(keys(snapshots)))

    # Select up to max_panels evenly spaced generations
    if length(gens) > max_panels
        idx  = round.(Int, range(1, length(gens), length = max_panels))
        gens = gens[idx]
    end
    n_panels = length(gens)

    ncols = min(3, n_panels)
    nrows = ceil(Int, n_panels / ncols)

    n_bits   = landscape.n_features
    all_fits = landscape.fitnesses
    fmin, fmax = Float64(minimum(all_fits)), Float64(maximum(all_fits))
    dataset_name = split(landscape.name, ".")[1]

    # Precompute hinged coords for local optima
    ox = [_hinged_coords(o, n_bits)[1] for o in local_optima]
    oy = [_hinged_coords(o, n_bits)[2] for o in local_optima]

    fig = Figure(size = figsize)
    Label(fig[0, 1:ncols],
        "$algorithm_name on $dataset_name — population snapshots",
        fontsize = 16, font = :bold)

    for (pi, g) in enumerate(gens)
        row = (pi - 1) ÷ ncols + 1
        col = (pi - 1) % ncols + 1

        ax = Axis(fig[row, col],
            title  = "gen $g",
            aspect = DataAspect()
        )
        hidedecorations!(ax)

        inds = snapshots[g]
        xs   = Float64[_hinged_coords(i, n_bits)[1] for i in inds]
        ys   = Float64[_hinged_coords(i, n_bits)[2] for i in inds]
        cs   = Float64[all_fits[i] for i in inds]

        scatter!(ax, xs, ys,
            color      = cs,
            colorrange = (fmin, fmax),
            colormap   = :viridis,
            markersize = 8,
            strokewidth = 0)

        if !isempty(local_optima)
            scatter!(ax, ox, oy,
                color = (:white, 0.5), markersize = 4, strokewidth = 0)
        end
    end

    # Shared colorbar
    Colorbar(fig[1:nrows, ncols + 1], colormap = :viridis,
        limits = (fmin, fmax), label = "Fitness")

    return fig
end