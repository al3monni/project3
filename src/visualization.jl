using CairoMakie

include("utils.jl")

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

function plot_evolution(history, title)
    generations = 1:size(history, 2)
    f = Figure(size = (900, 500))
    ax = Axis(
        f[1, 1],
        title = title,
        xlabel = "Generation",
        ylabel = "Fitness",
        limits = (0, GENERATIONS, 0.7, 1.0)
    )

    # Plot mean fitness
    lines!(ax, generations, history[3, :], label = "Mean Fitness", color=:blue)

    # Plot min/max as a shaded area
    band!(ax, generations, history[1, :], history[2, :], color=:blue, alpha=0.3, label="Min-Max Range")

    axislegend(ax)    
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

# 0.7 - 1.0
# 0.55 - 0.9
# 0 - 1.0