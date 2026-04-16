using EvoLP
using Statistics

function entropy(population::Vector{BitVector})
    n = length(population)
    if n == 0
        return 0.0
    end

    n_bits = length(population[1])
    bit_counts = zeros(Int, n_bits)

    for individual in population
        for j in 1:n_bits
            bit_counts[j] += individual[j] ? 1 : 0
        end
    end

    ent = 0.0
    for count in bit_counts
        p = count / n
        if p > 0 && p < 1
            ent -= p * log2(p) + (1 - p) * log2(1 - p)
        end
    end

    return ent
end

function compute!(history, fitnesses, population, generation)
    history[1, generation] = minimum(fitnesses)
    history[2, generation] = maximum(fitnesses)
    history[3, generation] = mean(fitnesses)
    history[4, generation] = std(fitnesses)
    history[5, generation] = entropy(population)
end

mutable struct Particle
    x::BitVector
    v::Vector{Float64}
    y::Float32
    x_best::BitVector
    y_best::Float32
end

sigmoid(z) = 1.0 / (1.0 + exp(-z))

function bitstring_to_index(x::BitVector)
    idx = 0
    for (i, bit) in enumerate(x)
        idx += bit ? 2^(i-1) : 0
    end
    return idx + 1  # +1 for 1-based indexing
end

function PSO!(
    landscape::Vector{Float32},
    popsize::Int64,
    k_max::Int64;
    w=1.0, c1=1.0, c2=1.0, maximize=true
    )
    L = length(landscape)
    n_bits = ceil(Int, log2(L))

    better = maximize ? (>) : (<)
    invalid_fitness = maximize ? 0.0 : 1.0

    population = [Particle(
        bitrand(n_bits),
        randn(n_bits),
        invalid_fitness,
        bitrand(n_bits),
        invalid_fitness
    ) for _ in 1:popsize]

    d = n_bits
    x_best = copy(population[1].x)
    y_best = invalid_fitness
    history = zeros(Float64, 5, k_max)

    runtime = @elapsed begin
        # initial evaluation
        for P in population
            idx = bitstring_to_index(P.x)
            P.y = idx <= L ? get_fitness(idx, landscape) : invalid_fitness

            P.x_best = copy(P.x)
            P.y_best = P.y

            if better(P.y, y_best)
                x_best = copy(P.x)
                y_best = P.y
            end
        end

        for gen in 1:k_max
            for P in population
                r1 = rand(d)
                r2 = rand(d)

                for j in 1:d
                    P.v[j] = w * P.v[j] +
                             c1 * r1[j] * (Int(P.x_best[j]) - Int(P.x[j])) +
                             c2 * r2[j] * (Int(x_best[j])   - Int(P.x[j]))
                    P.v[j] = clamp(P.v[j], -vmax, vmax)
                end

                for j in 1:d
                    P.x[j] = rand() < sigmoid(P.v[j])
                end

                idx = bitstring_to_index(P.x)
                P.y = idx <= L ? get_fitness(idx, landscape) : invalid_fitness

                if better(P.y, y_best)
                    x_best = copy(P.x)
                    y_best = P.y
                end

                if better(P.y, P.y_best)
                    P.x_best = copy(P.x)
                    P.y_best = P.y
                end
            end

            compute!(history, [P.y for P in population], [P.x for P in population], gen)
        end
    end

    best_i = maximize ? argmax([P.y_best for P in population]) :
                        argmin([P.y_best for P in population])

    best = population[best_i]
    n_evals = (1 + k_max) * length(population)

    return history, Result(best.y_best, best.x_best, population, k_max, n_evals, runtime)
end

# ==================== Genetic Algorithm ====================

function GA!(
    landscape::Vector{Float32},
    popsize::Int64,
    k_max::Int64;
    S::EvoLP.Selector=EvoLP.TournamentSelector(3),
    C::EvoLP.Recombinator=EvoLP.UniformCrossover(),
    M::EvoLP.Mutator=EvoLP.BitwiseMutator(0.05),
    pc=0.9,
    pm=-1.0
    )
    L = length(landscape)
    n_bits = ceil(Int, log2(L))

    # For maximization, invalid indices should be as bad as possible
    invalid_fitness = 0.0

    # Population of BitVectors
    population = binary_vector_pop(popsize, n_bits)

    # True fitness values (the objective we want to maximize)
    fitnesses = Vector{Float32}(undef, popsize)

    evaluate(x) = begin
        idx = bitstring_to_index(x)
        idx <= L ? get_fitness(idx, landscape) : invalid_fitness
    end

    # Initial evaluation
    fitnesses .= evaluate.(population)

    history = zeros(Float64, 5, k_max)

    runtime = @elapsed for gen in 1:k_max
        # EvoLP selectors minimize, so maximize f by minimizing -f
        sel_fitnesses = -fitnesses

        parents = [EvoLP.select(S, sel_fitnesses) for _ in eachindex(population)]

        offspring = Vector{typeof(population[1])}(undef, popsize)

        for i in eachindex(population)
            p1, p2 = parents[i]

            # crossover with probability pc
            child = if rand() < pc
                EvoLP.cross(C, population[p1], population[p2])
            else
                copy(population[rand((p1, p2))])
            end

            # mutation
            offspring[i] = if pm >= 0
                EvoLP.mutate(EvoLP.BitwiseMutator(pm), child)
            else
                EvoLP.mutate(M, child)
            end
        end

        # generational replacement
        population .= offspring

        # evaluate new population
        fitnesses .= evaluate.(population)

        # store min, max, mean, std of true fitnesses
        compute!(history, fitnesses, population, gen)
    end

    # Since we maximize, best is argmax
    best_i = argmax(fitnesses)
    best = population[best_i]
    n_evals = (k_max + 1) * popsize

    return history, Result(fitnesses[best_i], best, population, k_max, n_evals, runtime)
end