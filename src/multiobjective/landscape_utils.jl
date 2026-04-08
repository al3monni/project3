using HDF5
using Statistics

struct Landscape
    name::String
    mean_accuracies::Vector{Float64}
    mean_times::Vector{Float64}
    n_features::Int
end

function infer_n_features(num_rows::Int)
    n = round(Int, log2(num_rows + 1))
    if 2^n - 1 != num_rows
        error("El número de filas no corresponde a 2^n - 1")
    end
    return n
end

function load_landscape(filepath::String)
    h5open(filepath, "r") do f
        accuracies = read(f["accuracies"])
        times = read(f["times"])

        mean_accuracies = Float64.(vec(mean(accuracies, dims=2)))
        mean_times = Float64.(vec(mean(times, dims=2)))
        n_features = infer_n_features(length(mean_accuracies))

        return Landscape(filepath, mean_accuracies, mean_times, n_features)
    end
end

function bitvector_to_index(bits::BitVector)
    value = 0
    for b in bits
        value = (value << 1) | (b ? 1 : 0)
    end
    return value
end

function lookup_accuracy(bits::BitVector, landscape::Landscape)
    idx = bitvector_to_index(bits)
    if idx == 0
        return -1e9
    end
    return landscape.mean_accuracies[idx]
end

function lookup_time(bits::BitVector, landscape::Landscape)
    idx = bitvector_to_index(bits)
    if idx == 0
        return 1e9
    end
    return landscape.mean_times[idx]
end

function num_selected_features(bits::BitVector)
    return count(bits)
end

function evaluate_multiobjective(bits::BitVector, landscape::Landscape)
    nfeat = num_selected_features(bits)

    if nfeat == 0
        return (-1e9, -1e9)
    end

    acc = lookup_accuracy(bits, landscape)
    return (acc, -Float64(nfeat))
end