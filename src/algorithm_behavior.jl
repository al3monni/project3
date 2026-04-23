include("visualization.jl")

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

        arrows2d!(ax,
            [xs[i]], [ys[i]],
            [dx], [dy],
            color=color,
            tiplength=10,
            tipwidth=10
        )
    end
end # DEBUG OK

function plot_landscape_with_path(landscape, best_path, local_optima)

    # get the landscape (2D simple representation)
    f = plot_landscape(landscape, local_optima)

    # overlay the best path on the landscape
     overlay_path!(
        f[1,1],
        best_path,
        landscape[best_path]
    )

    return f
end # DEBUG OK

function plot_landscape_polar_with_path(landscape, best_path)

    # reuse base plot
    f = plot_landscape_polar(landscape)

    # recompute coordinates (same logic!)
    x, y = polar_coordinates(landscape)

    # overlay path using your existing function
    overlay_path!(
        f[1,1],
        x[best_path],
        y[best_path]
    )

    return f
end # DEBUG OK

function plot_hinged_map_with_path(landscape, best_path, local_optima)

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

    f = hinged_bitstring_map(landscape, local_optima)
    ax = f[1, 1]

    xs = x[best_path]
    ys = y[best_path]

    overlay_path!(ax, xs, ys)

    return f
end # DEBUG OK

function plot_lon_with_path(
    g,
    landscape,
    opt_index_map,
    basin_map,
    basin_sizes,
    best_path
    )

    # reuse base plot
    f, positions = plot_lon(
        g,
        landscape,
        opt_index_map,
        basin_sizes
    )

    node_path = [opt_index_map[basin_map[i]] for i in best_path]

    @show unique(best_path)
    @show unique(node_path)

    ax = f[1, 1]

    # extract coordinates from layout
    xs = [positions[i][1] for i in node_path]
    ys = [positions[i][2] for i in node_path]

    # overlay path
    overlay_path!(ax, xs, ys)

    return f
end # DEBUG OK