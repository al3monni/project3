using Random

struct Landscape
    name::String
    mean_accuracies::Vector{Float64}
    mean_times::Vector{Float64}
    n_features::Int
end

mutable struct Individual
    bits::BitVector
    objectives::Tuple{Float64, Float64}
    rank::Int
    crowding::Float64
end

function NSGA2!(
    landscape::Landscape,
    population::Vector{Individual},
    k_max::Integer;
    pc::Float64 = 0.9,
    pm::Float64 = -1.0,
    seed::Int = 42,
    verbose::Bool = true
)
    Random.seed!(seed)

    n_bits = landscape.n_features
    popsize = length(population)
    pm = pm < 0 ? 1.0 / n_bits : pm

    runtime = @elapsed begin
        for ind in population
            evaluate!(ind, landscape)
        end
        update_population_metadata!(population)

        for gen in 1:k_max
            offspring = create_offspring(population, popsize, pc, pm)

            for ind in offspring
                evaluate!(ind, landscape)
            end

            combined = vcat(population, offspring)
            population = environmental_selection(combined, popsize)
            update_population_metadata!(population)

            if verbose
                front = get_pareto_front(population)
                best_acc = maximum(ind.objectives[1] for ind in front)
                min_features = minimum(Int(-ind.objectives[2]) for ind in front)
                println(
                    "Gen $gen | Front 1: $(length(front)) solutions | " *
                    "Best acc: $(round(best_acc, digits = 6)) | " *
                    "Min features in F1: $min_features"
                )
            end
        end
    end

    pareto_front = unique_pareto_front(get_pareto_front(population))
    best = best_by_accuracy(pareto_front)
    n_evals = (1 + k_max) * popsize

    return (
        best_accuracy = best.objectives[1],
        best_solution = copy(best.bits),
        pareto_front = pareto_front,
        population = population,
        generations = k_max,
        n_evals = n_evals,
        runtime = runtime
    )
end

function infer_n_features(num_rows::Int)
    n = round(Int, log2(num_rows + 1))
    if 2^n - 1 != num_rows
        error("The number of rows does not match 2^n - 1.")
    end
    return n
end

function bitvector_to_index(bits::BitVector)
    value = 0
    for bit in bits
        value = (value << 1) | (bit ? 1 : 0)
    end
    return value
end

function lookup_accuracy(bits::BitVector, landscape::Landscape)
    idx = bitvector_to_index(bits)
    return idx == 0 ? -Inf : landscape.mean_accuracies[idx]
end

function num_selected_features(bits::BitVector)
    return count(bits)
end

function evaluate_multiobjective(bits::BitVector, landscape::Landscape)
    n_features = num_selected_features(bits)
    if n_features == 0
        return (-Inf, -Inf)
    end

    accuracy = lookup_accuracy(bits, landscape)
    return (accuracy, -Float64(n_features))
end

function evaluate!(ind::Individual, landscape::Landscape)
    ind.objectives = evaluate_multiobjective(ind.bits, landscape)
end

function repair_zero_features!(bits::BitVector)
    if count(bits) == 0
        bits[rand(1:length(bits))] = true
    end
    return bits
end

function random_individual(n_bits::Int)
    bits = BitVector(rand(Bool, n_bits))
    repair_zero_features!(bits)
    return Individual(bits, (0.0, 0.0), 0, 0.0)
end

function make_population(popsize::Int, n_bits::Int)
    return [random_individual(n_bits) for _ in 1:popsize]
end

function dominates(a::Individual, b::Individual)
    a1, a2 = a.objectives
    b1, b2 = b.objectives

    not_worse = (a1 >= b1) && (a2 >= b2)
    strictly_better = (a1 > b1) || (a2 > b2)

    return not_worse && strictly_better
end

function fast_non_dominated_sort!(population::Vector{Individual})
    n_pop = length(population)
    dominated_sets = [Int[] for _ in 1:n_pop]
    domination_counts = zeros(Int, n_pop)
    fronts = Vector{Vector{Int}}()
    first_front = Int[]

    for p in 1:n_pop
        for q in 1:n_pop
            p == q && continue

            if dominates(population[p], population[q])
                push!(dominated_sets[p], q)
            elseif dominates(population[q], population[p])
                domination_counts[p] += 1
            end
        end

        if domination_counts[p] == 0
            population[p].rank = 1
            push!(first_front, p)
        end
    end

    push!(fronts, first_front)

    i = 1
    while i <= length(fronts) && !isempty(fronts[i])
        next_front = Int[]
        for p in fronts[i]
            for q in dominated_sets[p]
                domination_counts[q] -= 1
                if domination_counts[q] == 0
                    population[q].rank = i + 1
                    push!(next_front, q)
                end
            end
        end

        if !isempty(next_front)
            push!(fronts, next_front)
        end
        i += 1
    end

    return fronts
end

function assign_crowding_distance!(population::Vector{Individual}, front::Vector{Int})
    isempty(front) && return nothing

    for i in front
        population[i].crowding = 0.0
    end

    if length(front) <= 2
        for i in front
            population[i].crowding = Inf
        end
        return nothing
    end

    n_objectives = 2

    for m in 1:n_objectives
        sorted_front = sort(front, by = i -> population[i].objectives[m])
        population[sorted_front[1]].crowding = Inf
        population[sorted_front[end]].crowding = Inf

        fmin = population[sorted_front[1]].objectives[m]
        fmax = population[sorted_front[end]].objectives[m]

        fmax == fmin && continue

        for j in 2:length(sorted_front)-1
            prev_obj = population[sorted_front[j - 1]].objectives[m]
            next_obj = population[sorted_front[j + 1]].objectives[m]

            if !isinf(population[sorted_front[j]].crowding)
                population[sorted_front[j]].crowding += (next_obj - prev_obj) / (fmax - fmin)
            end
        end
    end

    return nothing
end

function update_population_metadata!(population::Vector{Individual})
    fronts = fast_non_dominated_sort!(population)
    for front in fronts
        assign_crowding_distance!(population, front)
    end
    return fronts
end

function better(a::Individual, b::Individual)
    if a.rank < b.rank
        return true
    elseif a.rank > b.rank
        return false
    else
        return a.crowding > b.crowding
    end
end

function tournament_selection(population::Vector{Individual})
    i, j = rand(1:length(population), 2)
    a = population[i]
    b = population[j]
    return better(a, b) ? a : b
end

function uniform_crossover(parent1::BitVector, parent2::BitVector, pc::Float64)
    if rand() > pc
        return copy(parent1), copy(parent2)
    end

    child1 = copy(parent1)
    child2 = copy(parent2)

    for i in eachindex(parent1)
        if rand(Bool)
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        end
    end

    return child1, child2
end

function bitflip_mutation!(bits::BitVector, pm::Float64)
    for i in eachindex(bits)
        if rand() < pm
            bits[i] = !bits[i]
        end
    end
    return bits
end

function create_offspring(
    population::Vector{Individual},
    popsize::Int,
    pc::Float64,
    pm::Float64
)
    offspring = Individual[]

    while length(offspring) < popsize
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        child1_bits, child2_bits = uniform_crossover(parent1.bits, parent2.bits, pc)
        bitflip_mutation!(child1_bits, pm)
        bitflip_mutation!(child2_bits, pm)
        repair_zero_features!(child1_bits)
        repair_zero_features!(child2_bits)

        push!(offspring, Individual(child1_bits, (0.0, 0.0), 0, 0.0))
        if length(offspring) < popsize
            push!(offspring, Individual(child2_bits, (0.0, 0.0), 0, 0.0))
        end
    end

    return offspring
end

function environmental_selection(combined::Vector{Individual}, popsize::Int)
    fronts = fast_non_dominated_sort!(combined)
    new_population = Individual[]

    for front in fronts
        assign_crowding_distance!(combined, front)

        if length(new_population) + length(front) <= popsize
            append!(new_population, combined[front])
        else
            sorted_front = sort(combined[front], by = ind -> -ind.crowding)
            remaining = popsize - length(new_population)
            append!(new_population, sorted_front[1:remaining])
            break
        end
    end

    return new_population
end

function get_pareto_front(population::Vector{Individual})
    fronts = fast_non_dominated_sort!(population)
    return isempty(fronts) ? Individual[] : population[fronts[1]]
end

function unique_pareto_front(pareto_front::Vector{Individual})
    unique_dict = Dict{String, Individual}()

    for ind in pareto_front
        key = join(Int.(ind.bits))
        if !haskey(unique_dict, key)
            unique_dict[key] = ind
        end
    end

    return collect(values(unique_dict))
end

function best_by_accuracy(pareto_front::Vector{Individual})
    isempty(pareto_front) && error("The Pareto front is empty.")

    best = pareto_front[1]
    best_acc = best.objectives[1]

    for ind in pareto_front[2:end]
        if ind.objectives[1] > best_acc
            best = ind
            best_acc = ind.objectives[1]
        end
    end

    return best
end

function summarize_pareto_front(pareto_front::Vector{Individual})
    unique_front = unique_pareto_front(pareto_front)
    best = best_by_accuracy(unique_front)

    best_acc = best.objectives[1]
    min_features = minimum(Int(-ind.objectives[2]) for ind in unique_front)
    pareto_size = length(unique_front)

    return best_acc, min_features, pareto_size, unique_front
end