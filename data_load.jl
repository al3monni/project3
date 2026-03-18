using HDF5
using Statistics

# ============= Parameters =============

const dataset_name = "accuracies"
const datasets = [
    "01-breast-w_lr_F.h5",
    "05-credit-a_rf_F.h5",
    "08-letter-r_knn_F.h5"
]
const penalty = 0.01

# ============== Helpers ===============

count_ones(x::Integer) = count_ones(unsigned(x))

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

function triangle_fitness(x::Integer; m::Float32=1.0f0, s::Int=4)

    u = count_ones(x)

    # compute the block and position within the block
    block = (u - 1) ÷ s
    pos = (u - 1) % s

    if iseven(block)
        # even block, increasing fitness
        return m * (pos + 1)
    else
        # odd block, decreasing fitness
        return m * (s - pos)
    end
end

function triangle_lookup(n::Int; m::Float32=1.0f0, s::Int=4)

    size = 1 << n

    # preallocate the lookup table
    lookup = Vector{Float32}(undef, size)

    # precompute and store triangle fitness values
    @inbounds for x in 1:size
        lookup[x] = triangle_fitness(x; m=m, s=s)
    end

    return lookup
end

# =============== Main =================

function main()

    means = load_landscape(datasets[1])
    println("Prime 5 entità del vettore:")
    display(means[1:min(5, length(means))])
end

main()
