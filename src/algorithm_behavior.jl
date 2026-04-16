function overlay_path!(
    ax,
    xs::Vector{<:Real},
    ys::Vector{<:Real};
    color = :red
    )
    n = length(xs)

    # scatter points
    scatter!(ax, xs, ys, color=color, markersize=10)

    # draw arrows between consecutive points
    for i in 1:(n-1)
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]

        arrows!(ax,
            [xs[i]], [ys[i]],
            [dx], [dy],
            color=color,
            arrowsize=10,
            linewidth=2
        )
    end
end

function plot_landscape_with_path(fitness_lookup, best_path)
    f = plot_landscape(fitness_lookup)
    ax = f[1, 1]

    xs = best_path
    ys = fitness_lookup[best_path]

    overlay_path!(ax, xs, ys)

    return f
end

function hinged_map_with_path(landscape, best_path)
    n = length(landscape)
    bits = ceil(Int, log2(n))
    bits += isodd(bits)
    half = bits ÷ 2

    x = Int[]
    y = Int[]

    for i in 1:n
        b = lpad(string(i, base=2), bits, '0')
        push!(x, parse(Int, b[1:half]; base=2))
        push!(y, parse(Int, b[half+1:end]; base=2))
    end

    f = hinged_bitstring_map(landscape)
    ax = f[1, 1]

    xs = x[best_path]
    ys = y[best_path]

    overlay_path!(ax, xs, ys)

    return f
end

function overlay_lon_path!(
    ax,
    layout_positions,
    best_path,
    basin_map,
    opt_index_map;
    color = :red
    )
    # map individuals → node indices
    node_path = [
        opt_index_map[basin_map[i]]
        for i in best_path
    ]

    xs = [layout_positions[i][1] for i in node_path]
    ys = [layout_positions[i][2] for i in node_path]

    overlay_path!(ax, xs, ys; color=color)
end

function plot_lon_with_path(
    g,
    landscape,
    opt_index_map,
    basin_sizes,
    basin_map,
    best_path;
    layout_algo = :spring
    )

    f = plot_lon(g, landscape, opt_index_map, basin_sizes;
                 layout_algo=layout_algo)

    ax = f[1, 1]

    # recompute layout (same as plot_lon!)
    layout = layout_algo == :spring ? Spring() :
             layout_algo == :kamada ? KamadaKawai() :
             Spectral()

    positions = layout(g)

    overlay_lon_path!(
        ax,
        positions,
        best_path,
        basin_map,
        opt_index_map
    )

    return f
end