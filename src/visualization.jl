using CairoMakie

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
    f
end