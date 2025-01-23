import numpy as np
from general_functions import get_cities, plot_final_path, plot_loss_history

def initialize_population(cities, pop_size=None):
    num_cities = cities.shape[0]

    if pop_size is None:
        base_size = 10 * num_cities
        pop_size = max(50, min(base_size, 5000))  # between 50-5000
        print(f"Auto-set population size to {pop_size} based on {num_cities} cities")
    
    return np.array([np.random.permutation(num_cities) for _ in range(pop_size)])

def create_distance_matrix(cities):
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis = 2))

def calculate_fitness(route, distance_matrix):
    closed_route = np.append(route, route[0])

    total_distance = np.sum(
        distance_matrix[
            closed_route[:-1],
            closed_route[1:]
        ]
    )

    return 1 / total_distance

def get_fittest_routes(population, distance_matrix, k):
    fitness_scores = np.array([calculate_fitness(route, distance_matrix) 
                               for route in population])
    top_k_indices = np.argsort(fitness_scores)[-k:][::-1]
    return population[top_k_indices]

def one_point_crossover(parent_a, parent_b):
    n = len(parent_a)
    cross_point = np.random.randint(1, n // 2 + 1)  # 0 <= cross <= n

    child_part = parent_a[:cross_point + 1]

    mask = ~np.isin(parent_b, child_part)
    remaining = parent_b[mask]

    child = np.concatenate([child_part, remaining])

    assert len(np.unique(child)) == n, "Invalid child permutation"
    return child

def fitness_proportional_pairing(population, fitness_scores, num_pairs):
    probs = fitness_scores / np.sum(fitness_scores)

    parent_indices = np.random.choice(
        len(population),
        size=num_pairs * 2,
        p=probs,
        replace=True
    )

    pairs = parent_indices.reshape(num_pairs, 2)

    return [(population[a], population[b]) for a, b in pairs]

def genetic_algorithm(cities, num_iter=1000, num_pop=None, num_fittest=2, 
                      early_stopping=True, stopping_threshold=1e-5, verbose=True):

    pop = initialize_population(cities, num_pop)
    if num_pop == None:
        num_pop = len(pop)
    distance_matrix = create_distance_matrix(cities)
    history = []
    best_distance = np.inf
    
    for iter in range(num_iter):

        fitness = np.array([calculate_fitness(route, distance_matrix) 
                           for route in pop])

        current_best_idx = np.argmax(fitness)
        current_best_distance = 1 / fitness[current_best_idx]
        history.append(current_best_distance)

        if early_stopping and iter > 10:
            improvement = best_distance - current_best_distance
            if improvement < stopping_threshold:
                if verbose:
                    print(f"Early stopping at iteration {iter} (improvement < {stopping_threshold})")
                break

        if current_best_distance < best_distance:
            best_distance = current_best_distance

        if verbose and (iter % 10 == 0 or iter == num_iter - 1):
            print(f"Iter {iter:4d} | Best Distance: {current_best_distance:.4f}")
        
        #select fittest parents
        sorted_indices = np.argsort(fitness)[::-1]
        elites = pop[sorted_indices[:num_fittest]]
        
        #fittest children
        num_children_needed = num_pop - num_fittest
        fittest_children = []
        min_child_fitness = -np.inf

        #select parent pairs with proportional probs
        num_pairs = num_children_needed * 2
        parent_pairs = fitness_proportional_pairing(pop, fitness, num_pairs)

        for parent_a, parent_b in parent_pairs:
            child = one_point_crossover(parent_a, parent_b)
            child_fitness = calculate_fitness(child, distance_matrix)
            
            #save top children
            if len(fittest_children) < num_children_needed:
                fittest_children.append(child)
                min_child_fitness = min(min_child_fitness, child_fitness)
            elif child_fitness > min_child_fitness:
                #replace weakest child
                weakest_idx = np.argmin([calculate_fitness(c, distance_matrix)
                                        for c in fittest_children])
                fittest_children[weakest_idx] = child
                min_child_fitness = min(calculate_fitness(c, distance_matrix) 
                                       for c in fittest_children)
        
        #create new population
        pop = np.vstack([elites, np.array(fittest_children)])
    
    #return best route and history
    final_fitness = np.array([calculate_fitness(route, distance_matrix) 
                             for route in pop])
    best_route = pop[np.argmax(final_fitness)]
    return best_route, history

# cities = get_cities("tsp_cities.csv", use_existing=False, num_cities=20)

# best_route, history = genetic_algorithm(
#     cities=cities,
#     num_iter=1000,
#     num_pop=100,
#     num_fittest=3,
#     early_stopping=True,
#     stopping_threshold=1e-4,
#     verbose=True
# )

# plot_loss_history(history)
# plot_final_path(cities, best_route)