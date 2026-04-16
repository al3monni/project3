using Combinatorics

function get_local_optima(landscape::Vector{Float32}; k::Int=1)

    """ compute the set of local optima for a given landscape """
    
    n = length(landscape)
    bits = ceil(Int, log2(n))
    
    local_optima = Int[]
    
    for i in 1:n
        # Check if current fitness is >= all neighbors
        if all(landscape[i] >= landscape[j] for j in neighbors(i, bits; k=k))
            push!(local_optima, i)
        end
    end
    
    return local_optima
end

function neighbors(index::Int, n_bits::Int; k::Int=1)
    neigh = Int[]

    for d in 1:k
        for positions in combinations(0:n_bits-1, d)
            mask = 0
            for i in positions
                mask |= (1 << i)
            end

            neighbor = index ⊻ mask

            if neighbor != 0
                push!(neigh, neighbor)
            end
        end
    end

    return neigh
end

function to_bitstring(x::Int, n_bits::Int)
    s = string(x, base=2)
    return lpad(s, n_bits, '0')
end

function main_old()
    n_bits = 4
    n = 0
    neigh = neighbors_param(n, n_bits; k=2)

    println(n, " = ", to_bitstring(n, n_bits))

    for i in neigh
        println(to_bitstring(i, n_bits))
    end
end
