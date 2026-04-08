
function get_local_optima(landscape::Vector{Float32})
    n = length(landscape)
    bits = ceil(Int, log2(n))

    local_optima = Int[]

    for i in 1:n

        neighbors = get_neighbors(i, bits)

        best = true
        for neighbor in neighbors
            if landscape[neighbor] > landscape[i]
                best = false
                break
            end
        end

        if best
            push!(local_optima, i)
        end
    end

    return local_optima
end


function get_neighbors(index::Int, n_bits::Int)

    neigh = Int[]

    for i in 0:n_bits-1

        neighbor = index ⊻ (1 << i)  # flip bit i

        if neighbor == 0
            continue
        else 
            push!(neigh, neighbor)
        end
    end

    return neigh
end