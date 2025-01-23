import numpy as np
import matplotlib.pyplot as plt

def get_cities(filename, use_existing, num_cities=10, x_range=(0,100), y_range=(0,100)):
    if use_existing:
        try:
            data = np.genfromtxt(filename, delimiter=',', skip_header=1)
            return data[:, 1:]
        except OSError:
            raise ValueError("Cities file not found. Please create it first")
    else:
        cities = np.column_stack((
            np.random.uniform(x_range[0], x_range[1], num_cities),
            np.random.uniform(y_range[0], y_range[1], num_cities)
        ))
        
        header = "city,x,y"
        indices = np.arange(num_cities).reshape(-1, 1)
        np.savetxt(filename, np.hstack((indices, cities)), 
                 delimiter=',', header=header, comments='', fmt=['%d', '%f', '%f'])
        return cities

def plot_loss_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Best Distance")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.title("Min Distance Over Time")
    plt.grid()
    plt.legend()
    plt.show()

def plot_final_path(cities, best_route):
    ordered_cities = cities[best_route]
    closed_route = np.vstack([ordered_cities, ordered_cities[0]])
    
    plt.figure(figsize=(10, 6))
    plt.plot(closed_route[:, 0], closed_route[:, 1], 'o-', label="Path", alpha=.5)
    plt.scatter(cities[:, 0], cities[:, 1], c='black', label="Cities")

    start_city = ordered_cities[0]
    plt.scatter(start_city[0], start_city[1], c='green', s=150, label="Start", edgecolor='black', zorder=5)

    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f"{i}", fontsize=12, ha='right')
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Final TSP Path")
    plt.grid(True)
    plt.legend()
    plt.show()