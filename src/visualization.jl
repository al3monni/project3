using CairoMakie

function hinged_bitstring_map(landscape::Vector{Float32})
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

    f = Figure()
    ax = Axis(f[1, 1],
        title = "Hinged Bitstring Map",
        xlabel = "First half",
        ylabel = "Second half"
    )
    # map = scatter!(ax, x, y,
    #     color = landscape,
    #     colormap = :viridis
    # )
    map = heatmap!(ax, x, y, landscape,
        colormap = :viridis
    )
    Colorbar(f[1, 2], map, label = "Fitness")
    f
end