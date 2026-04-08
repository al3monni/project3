include("landscape_utils.jl")
using Random

# =========================================================
# ESTRUCTURA DEL INDIVIDUO
# =========================================================

mutable struct Individual
    bits::BitVector
    objectives::Tuple{Float64, Float64}   # (accuracy, -num_features)
    rank::Int
    crowding::Float64
end

# =========================================================
# CREACIÓN Y EVALUACIÓN
# =========================================================

function random_individual(n_bits::Int)
    bits = BitVector(rand(Bool, n_bits))
    repair_zero_features!(bits)
    return Individual(bits, (0.0, 0.0), 0, 0.0)
end

function repair_zero_features!(bits::BitVector)
    if count(bits) == 0
        bits[rand(1:length(bits))] = true
    end
end

function evaluate!(ind::Individual, landscape::Landscape)
    ind.objectives = evaluate_multiobjective(ind.bits, landscape)
end

# =========================================================
# DOMINANCIA
# =========================================================

function dominates(a::Individual, b::Individual)
    a1, a2 = a.objectives
    b1, b2 = b.objectives

    not_worse = (a1 >= b1) && (a2 >= b2)
    strictly_better = (a1 > b1) || (a2 > b2)

    return not_worse && strictly_better
end

# =========================================================
# FAST NON-DOMINATED SORT
# =========================================================

function fast_non_dominated_sort!(population::Vector{Individual})
    N = length(population)

    S = [Int[] for _ in 1:N]
    n = zeros(Int, N)
    fronts = Vector{Vector{Int}}()
    first_front = Int[]

    for p in 1:N
        for q in 1:N
            if p == q
                continue
            end

            if dominates(population[p], population[q])
                push!(S[p], q)
            elseif dominates(population[q], population[p])
                n[p] += 1
            end
        end

        if n[p] == 0
            population[p].rank = 1
            push!(first_front, p)
        end
    end

    push!(fronts, first_front)

    i = 1
    while i <= length(fronts) && !isempty(fronts[i])
        next_front = Int[]
        for p in fronts[i]
            for q in S[p]
                n[q] -= 1
                if n[q] == 0
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

# =========================================================
# CROWDING DISTANCE
# =========================================================

function assign_crowding_distance!(population::Vector{Individual}, front::Vector{Int})
    if isempty(front)
        return
    end

    for i in front
        population[i].crowding = 0.0
    end

    if length(front) <= 2
        for i in front
            population[i].crowding = Inf
        end
        return
    end

    num_objectives = 2

    for m in 1:num_objectives
        sorted_front = sort(front, by = i -> population[i].objectives[m])

        population[sorted_front[1]].crowding = Inf
        population[sorted_front[end]].crowding = Inf

        fmin = population[sorted_front[1]].objectives[m]
        fmax = population[sorted_front[end]].objectives[m]

        if fmax == fmin
            continue
        end

        for j in 2:length(sorted_front)-1
            prev_obj = population[sorted_front[j - 1]].objectives[m]
            next_obj = population[sorted_front[j + 1]].objectives[m]

            if !isinf(population[sorted_front[j]].crowding)
                population[sorted_front[j]].crowding += (next_obj - prev_obj) / (fmax - fmin)
            end
        end
    end
end

# =========================================================
# SELECCIÓN
# =========================================================

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

# =========================================================
# OPERADORES GENÉTICOS
# =========================================================

function uniform_crossover(parent1::BitVector, parent2::BitVector, pc::Float64)
    n = length(parent1)

    if rand() > pc
        return copy(parent1), copy(parent2)
    end

    child1 = copy(parent1)
    child2 = copy(parent2)

    for i in 1:n
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
end

# =========================================================
# OFFSPRING
# =========================================================

function create_offspring(population::Vector{Individual}, popsize::Int, pc::Float64, pm::Float64)
    offspring = Individual[]

    while length(offspring) < popsize
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)

        c1_bits, c2_bits = uniform_crossover(p1.bits, p2.bits, pc)

        bitflip_mutation!(c1_bits, pm)
        bitflip_mutation!(c2_bits, pm)

        repair_zero_features!(c1_bits)
        repair_zero_features!(c2_bits)

        push!(offspring, Individual(c1_bits, (0.0, 0.0), 0, 0.0))
        if length(offspring) < popsize
            push!(offspring, Individual(c2_bits, (0.0, 0.0), 0, 0.0))
        end
    end

    return offspring
end

# =========================================================
# REEMPLAZO NSGA-II
# =========================================================

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

# =========================================================
# NSGA-II PRINCIPAL
# =========================================================

function nsga2(landscape::Landscape;
               popsize::Int = 100,
               generations::Int = 50,
               pc::Float64 = 0.9,
               pm::Float64 = -1.0,
               seed::Int = 42)

    Random.seed!(seed)

    n_bits = landscape.n_features

    if pm < 0
        pm = 1.0 / n_bits
    end

    population = [random_individual(n_bits) for _ in 1:popsize]

    for ind in population
        evaluate!(ind, landscape)
    end

    fronts = fast_non_dominated_sort!(population)
    for front in fronts
        assign_crowding_distance!(population, front)
    end

    for gen in 1:generations
        offspring = create_offspring(population, popsize, pc, pm)

        for ind in offspring
            evaluate!(ind, landscape)
        end

        combined = vcat(population, offspring)
        population = environmental_selection(combined, popsize)

        fronts = fast_non_dominated_sort!(population)
        for front in fronts
            assign_crowding_distance!(population, front)
        end

        front1 = population[fronts[1]]
        best_acc = maximum(ind.objectives[1] for ind in front1)
        min_features = minimum(Int(-ind.objectives[2]) for ind in front1)

        println("Gen $gen | Frente 1: $(length(front1)) soluciones | Mejor acc: $(round(best_acc, digits=6)) | Min features en F1: $min_features")
    end

    final_fronts = fast_non_dominated_sort!(population)
    for front in final_fronts
        assign_crowding_distance!(population, front)
    end

    pareto_front = population[final_fronts[1]]
    return population, pareto_front
end

# =========================================================
# IMPRESIÓN DEL FRENTE DE PARETO
# =========================================================

function print_pareto_front(pareto_front::Vector{Individual}; max_show::Int = 20)
    println("\n===== FRENTE DE PARETO (SIN DUPLICADOS) =====")

    unique_dict = Dict{String, Individual}()

    for ind in pareto_front
        key = join(Int.(ind.bits))
        if !haskey(unique_dict, key)
            unique_dict[key] = ind
        end
    end

    unique_front = collect(values(unique_dict))

    sorted_pf = sort(
        unique_front,
        by = ind -> (ind.objectives[1], ind.objectives[2]),
        rev = true
    )

    limit = min(max_show, length(sorted_pf))

    println("Número de soluciones únicas en Pareto: ", length(sorted_pf))

    for i in 1:limit
        ind = sorted_pf[i]
        acc = ind.objectives[1]
        nfeat = Int(-ind.objectives[2])
        bits_str = join(Int.(ind.bits))
        println("Solución $i | acc = $(round(acc, digits=6)) | n_features = $nfeat | bits = $bits_str")
    end
end