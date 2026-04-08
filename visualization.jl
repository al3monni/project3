using CairoMakie

function plot_landscape(fitness_lookup::Vector{Float32})

    # extract the individuals
    x = eachindex(fitness_lookup)

    fig = Figure(size = (900, 500))
    
    ax = Axis(
        fig[1, 1],
        title = "Fitness Landscape",
        xlabel = "Individual index",
        ylabel = "Fitness"
    )

    lines!(ax, x, fitness_lookup)
    scatter!(ax, x, fitness_lookup)

    return fig
end

function plot_landscape_polar(
    fitness_lookup::Vector{Float32};
    base_radius::Real = 0.1,
    radial_scale::Real = 0.4,
    markersize::Real = 6,
    show_points::Bool = true,
    show_line::Bool = true
)
    n = length(fitness_lookup)
    f = Float64.(fitness_lookup)

    # Normalize fitness to [0, 1]
    fmin = minimum(f)
    fmax = maximum(f)

    fnorm = if fmax == fmin
        fill(0.5, n)
    else
        (f .- fmin) ./ (fmax - fmin)
    end

    # Radius = base circle + amplified variation
    r = base_radius .+ radial_scale .* fnorm

    θ = range(0, 2π, length = n + 1)[1:end-1]

    x = r .* cos.(θ)
    y = r .* sin.(θ)

    fig = Figure(size = (800, 800))
    ax = Axis(
        fig[1, 1],
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

    return fig
end