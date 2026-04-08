
function get_local_optima(landscape::Vector{Float32})
    n = length(landscape)
    bits = ceil(Int, log2(n))

    local_optima = Int[]

    for i in 1:n
        b = lpad(string(i, base=2), bits, '0')

        neighbors = get_neighbors(b)

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

function get_neighbors(individual::String)
    n = length(individual)
    neighbors = Int[]

    for i in 1:n
        # flip the i-th bit
        flipped = individual[i] == '0' ? '1' : '0'

        neighbor = individual[1:i-1] * flipped * individual[i+1:end]
        neighbor = parse(Int, neighbor; base=2)

        if neighbor == 0
            continue
        end
        push!(neighbors, neighbor)
    end

    return neighbors
end