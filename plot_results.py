import os
import numpy as np
import matplotlib.pyplot as plt

def plot_average_and_max_fitness_history(average_fitness_history: list, max_fitness_history: list) -> None:
    os.makedirs("results", exist_ok=True)

    plt.plot(average_fitness_history, marker='o', label='Average Fitness')
    plt.plot(max_fitness_history, marker='x', label='Max Fitness')
    plt.title("Average and Max Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/average_and_max_fitness_history.png")
    plt.show()
    plt.close()
