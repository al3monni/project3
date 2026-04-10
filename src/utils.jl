function get_local_optima(landscape::Vector{Float32})

    """ compute the set of local optima for a given landscape """
    
    n = length(landscape)
    bits = ceil(Int, log2(n))
    
    local_optima = Int[]
    
    for i in 1:n
        # Check if current fitness is >= all neighbors
        if all(landscape[i] >= landscape[j] for j in neighbors(i, bits))
            push!(local_optima, i)
        end
    end
    
    return local_optima
end

function neighbors(index::Int, n_bits::Int)

    """ optimized function to generate the neighbours of a given individual """

    neigh = Int[]

    for i in 0:n_bits-1

        neighbor = index ⊻ (1 << i)  # flip bit i

        if neighbor != 0
            push!(neigh, neighbor)  
        end
    end

    return neigh
end