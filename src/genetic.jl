using EvoLP

struct Particle
    x,
    v,
    y,
    x_best,
    y_best
end

function PSO!(
    logger::Logbook, f::Function, population::Vector{Particle}, k_max::Integer;
    w=1, c1=1, c2=1
)
    d = length(population[1].x)
    x_best, y_best = copy(population[1].x_best), Inf

    # evaluation loop
    runtime = @elapsed begin
        for P in population
            P.y = f(P.x)  # O(pop)

            if P.y < y_best
                x_best[:] = P.x
                y_best = P.y
            end
        end

        for _ in 1:k_max
            for P in population
                r1, r2 = rand(d), rand(d)
                P.x += P.v
                P.v = w * P.v + c1 * r1 .* (P.x_best - P.x) + c2 * r2 .* (x_best - P.x)
                P.y = f(P.x)  # O(k_max * pop)

                if P.y < y_best
                    x_best[:] = P.x
                    y_best = P.y
                end

                if P.y < P.y_best
                    P.x_best[:] = P.x
                    P.y_best = P.y
                end
            end
            compute!(logger, [P.y for P in population])
        end
    end

    best_i = argmin([P.y_best for P in population])
    best = population[best_i]
    n_evals = (1 + k_max) * length(population)

    return Result(best.y_best, best.x_best, population, k_max, n_evals, runtime)
end