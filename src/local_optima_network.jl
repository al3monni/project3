using Graphs
using GraphPlot
using CairoMakie
using GraphMakie
using NetworkLayout
using CSV, DataFrames

include("utils.jl")

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
