using Graphs
using CairoMakie
using NetworkLayout

include("utils.jl")

# --- Hill climbing to find basin attractor
function hill_climb(start::Int, fitness::Vector{Float64}, n_bits::Int)
    current = start
    while true
        neigh = get_neighbors(current, n_bits)
        best = current
        for n in neigh
            if n < length(fitness) && fitness[n+1] > fitness[best+1]
                best = n
            end
        end
        if best == current
            return current
        end
        current = best
    end
end

# --- Assign each state to a basin (local optimum)
function compute_basins(fitness::Vector{Float64}, n_bits::Int)
    basin_map = Dict{Int, Int}()
    for i in 0:length(fitness)-1
        basin_map[i] = hill_climb(i, fitness, n_bits)
    end
    return basin_map
end

function build_lon(
    fitness::Vector{Float64},
    local_optima::Vector{Int},
    n_bits::Int
    )

    basin_map = compute_basins(fitness, n_bits)

    # Map optima to node ids
    opt_index = Dict(opt => i for (i, opt) in enumerate(local_optima))

    g = DiGraph(length(local_optima))

    weights = Dict{Tuple{Int,Int}, Float64}()

    for state in 0:length(fitness)-1

        b1 = basin_map[state]

        for neigh in neighbors(state, n_bits)

            if neigh < length(fitness)

                b2 = basin_map[neigh]

                if b1 != b2 && haskey(opt_index, b1) && haskey(opt_index, b2)
                    u = opt_index[b1]
                    v = opt_index[b2]
                    weights[(u,v)] = get(weights, (u,v), 0.0) + 1.0
                end
            end
        end
    end

    # Add edges
    for ((u,v), w) in weights
        add_edge!(g, u, v)
    end

    return g, weights
end

function plot_lon_makie(g::DiGraph, weights; node_labels=true)
    n = nv(g)

    # --- Layout (spring layout)
    layout = spring(g)  # from NetworkLayout.jl
    xs = [p[1] for p in layout]
    ys = [p[2] for p in layout]

    fig = Figure()
    ax = Axis(fig[1,1], title="Local Optima Network")

    # --- Draw edges
    for e in edges(g)
        u, v = src(e), dst(e)

        x1, y1 = xs[u], ys[u]
        x2, y2 = xs[v], ys[v]

        w = get(weights, (u,v), 1.0)

        # thickness scaled by weight
        lines!(ax, [x1, x2], [y1, y2], linewidth=1 + 2w)

        # edge label (midpoint)
        mx, my = (x1+x2)/2, (y1+y2)/2
        text!(ax, string(round(w, digits=2)), position=(mx, my), align=(:center, :center))
    end

    # --- Draw nodes
    scatter!(ax, xs, ys, markersize=20)

    # --- Node labels
    if node_labels
        for i in 1:n
            text!(ax, string(i), position=(xs[i], ys[i]),
                  align=(:center, :center), offset=(0, 10))
        end
    end

    fig
end

#-------------------------------

function compute_basins_wrong(
    fitness::Vector{Float64},
    local_optima::Vector{Int}
    )

    # Local function to compute the hamming distance
    hamming_dist(a, b) = count_ones(a ⊻ b)

    # Preallocate the basin map
    basin_map = Dict{Int, Int}()

    # For each bitstring find the closest local optimum
    for i in 0:length(fitness)-1
        # find the closest local optimum in terms of hamming distance
        hamming_min = typemax(Int)
        optimal = -1
        for opt in local_optima
            dist = hamming_dist(i, opt)
            if dist < hamming_min
                hamming_min = dist
                optimal = opt
            end
        end
        basin_map[i] = optimal
    end

    return basin_map
end
