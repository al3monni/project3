using HDF5
using Statistics

# ============= Landscape and Fitness Functions =============

function load_landscape(filename::String)

    # obtain a reference for the hdf5 file
    h5open("train/"*filename, "r") do f

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

# ==================== Triangle Function ====================

function triangle_function(b::Integer; m::Float32=1.0f0, s::Int=4)

    r = abs(b)
    t = mod(r, 2s)

    if t <= s
        return m * t
    else
        return m * (2s - t)
    end
end

function triangle_landscape(n::Integer; m::Float32=1.0f0, s::Int=4)

    # preallocate the lookup table
    lookup = Vector{Float32}(undef, n)

    # precompute and store triangle fitness values
    @inbounds for x in 1:n                              # pay attention here
        lookup[x] = triangle_function(x; m=m, s=s)
    end

    return lookup
end