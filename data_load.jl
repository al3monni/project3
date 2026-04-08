using HDF5
using Statistics
include("visualization.jl")

# ============= Parameters =============

const dataset_name = "accuracies"
const datasets = [
    "01-breast-w_lr_F.h5",
    "05-credit-a_rf_F.h5",
    "08-letter-r_knn_F.h5"
]
const penalty = 0.01
const n = 16
const m = 1.0f0
const s = 4

# ============== Helpers ===============

count_ones(x::Integer) = Base.count_ones(unsigned(x))

# ============= Functions ==============

function load_landscape(filename::String)
    # obtain a reference for the hdf5 file
    h5open("train data/"*filename, "r") do f
        # select the "accuracies" dataset
        data = read(f[dataset_name])
        # compute the mean of each row
        return vec(mean(data, dims=2))
    end
end

function get_fitness(x::Integer, lookup::Vector{Float32}, w::Real)
    # compute the penalty based on feature number
    penalty = w * count_ones(x)
    # return the penalized fitness
    return lookup[x] - penalty
end

function triangle_fitness(b::Integer; m::Float32=1.0f0, s::Int=4)

    r = abs(b)
    t = mod(r, 2s)

    if t <= s
        return m * t
    else
        return m * (2s - t)
    end
end

function triangle_lookup(n::Integer; m::Float32=1.0f0, s::Int=4)

    # preallocate the lookup table
    lookup = Vector{Float32}(undef, n)

    # precompute and store triangle fitness values
    @inbounds for x in 1:n                              # pay attention here
        lookup[x] = triangle_fitness(x; m=m, s=s)
    end

    return lookup
end

# =============== Main =================

function main()

    landscape = load_landscape(datasets[1])
    #landscape = triangle_lookup(n; m=m, s=s)
    fig = plot_landscape_polar(landscape)
    display(fig)

end

main()
