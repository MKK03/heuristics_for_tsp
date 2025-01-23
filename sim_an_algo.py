import numpy as np
from general_functions import get_cities, plot_final_path, plot_loss_history

def generate_initial_route(cities):
    return np.random.permutation(len(cities))

def calculate_total_distance(route, cities):
    total = 0
    n = len(route)
    for i in range(n):
        current_city = cities[route[i]]
        next_city = cities[route[(i + 1) % n]]
        dx = next_city[0] - current_city[0]
        dy = next_city[1] - current_city[1]
        total += np.sqrt(dx**2 + dy**2)
    return total

def generate_neighbor_route(current_route):
    new_route = current_route.copy()
    i, j = np.random.choice(len(new_route), size=2, replace=False)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def acceptance_probability(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return 1.0
    else:
        return np.exp(-(new_cost - current_cost) / temperature)

def annealing_schedule(initial_temp, cooling_rate, min_temp):
    current_temp = initial_temp
    while current_temp > min_temp:
        yield current_temp
        current_temp *= cooling_rate

def simulated_annealing(cities, initial_temp = 10000, cooling_rate = 0.95,
                        min_temp = 1e-5, verbose = True, early_stopping = True, patience = 10):
    iteration = 0
    last_improvement = 0
    current_route = generate_initial_route(cities)
    current_cost = calculate_total_distance(current_route, cities)
    best_route = current_route.copy()
    best_cost = current_cost
    history = [current_cost]

    schedule = annealing_schedule(initial_temp, cooling_rate, min_temp)

    for temperature in schedule:
        iteration += 1
        neighbor_route = generate_neighbor_route(current_route)
        neighbor_cost = calculate_total_distance(neighbor_route, cities)

        if acceptance_probability(current_cost, neighbor_cost, temperature) > np.random.random():
            current_route = neighbor_route
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_route = current_route.copy()
                best_cost = current_cost
                last_improvement = iteration

        history.append(current_cost)

        if verbose and iteration % 10 == 0:
            
            print(f"Temp: {temperature:.2f} | Current Cost: {current_cost:.2f} | Best Cost: {best_cost:.2f} | Iteration number: {iteration}")
        
        if early_stopping and (iteration - last_improvement) >= patience:
            if verbose:
                print(f"\nEarly stopping: No improvement for {patience} iterations.")
            break

    return best_route, np.array(history)

# cities = get_cities("tsp_cities.csv", use_existing=False, num_cities=20)

# best_route, history = simulated_annealing(cities)

# plot_loss_history(history)
# plot_final_path(cities, best_route)