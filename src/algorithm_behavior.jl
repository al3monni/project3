function overlay_path!(
    ax,
    xs::Vector{<:Real},
    ys::Vector{<:Real};
    color = :red
    )

    n = length(xs)
    scatter!(ax, xs, ys, color=color, markersize=10)

    for i in 1:(n-1)
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        arrows!(ax, [xs[i]], [ys[i]], [dx], [dy],
            color=color, arrowsize=10)
    end
end

function plot_landscape_with_path(landscape::Landscape, best_path, local_optima)
    f = plot_landscape(landscape.fitnesses, local_optima)
    overlay_path!(f[1,1], Float64.(best_path), Float64.(landscape.fitnesses[best_path]))
    return f
end

function plot_landscape_polar_with_path(landscape::Landscape, best_path)
    f = plot_landscape_polar(landscape.fitnesses)
    x, y = polar_coordinates(landscape.fitnesses)
    overlay_path!(f[1,1], x[best_path], y[best_path])
    return f
end

function plot_hinged_map_with_path(landscape::Landscape, best_path, local_optima)
    n = length(landscape.fitnesses)
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

    f = hinged_bitstring_map(landscape.fitnesses, local_optima)
    overlay_path!(f[1,1], Float64.(x[best_path]), Float64.(y[best_path]))
    return f
end

function plot_lon_with_path(
    g,
    landscape::Landscape,
    opt_index_map,
    basin_map,
    basin_sizes,
    best_path
    )

    f, positions = plot_lon(g, landscape.fitnesses, opt_index_map, basin_sizes)

    node_path = [opt_index_map[basin_map[i]] for i in best_path]
    ax = f[1, 1]
    xs = [positions[i][1] for i in node_path]
    ys = [positions[i][2] for i in node_path]
    overlay_path!(ax, xs, ys)
    return f
end