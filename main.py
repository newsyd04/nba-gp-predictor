import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from config import PARAMETERS, TABLE_PATH
from data_loader import load_nba_data
from population import make_population
from fitness import fitness_worker, average_fitness, max_fitness, get_predictions, get_population_sorted_by_fitness
from selection.roulette import RouletteSelection
from evolve.crossover import run_crossover
from evolve.mutation import run_mutations
from plot_results import plot_average_and_max_fitness_history

if __name__ == "__main__":
    t_setup_start = time.time()
    variables, train_data_raw, test_data_raw = load_nba_data(TABLE_PATH)
    population = make_population(
        PARAMETERS["population_size"],
        PARAMETERS["max_tree_height"],
        variables
    )
    # removing the target column from train and test
    train_target_outputs = [row[-1] for row in train_data_raw]
    train_data = [row[:-1] for row in train_data_raw]
    test_data = [row[:-1] for row in test_data_raw]
    test_targets = [row[-1] for row in test_data_raw]
    average_fitness_history = []
    max_fitness_history = []
    t_setup_end = time.time()
    print(f"Data loading and population initialization took {t_setup_end - t_setup_start:.2f} seconds.")

    for generation in range(1, PARAMETERS["max_generations"] + 1):
        print(f"Generation {generation}")
        t_gen_start = time.time()

        # compute fitness scores
        t_fit_start = time.time()
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(
                executor.map(
                    fitness_worker,
                    [(individual, train_target_outputs, train_data, variables) for individual in population]
                )
            )
        fitness_scores, prediction_times, loss_times = zip(*results)
        t_fit_end = time.time()
        print(f"Total fitness time without concurrency: {(sum(prediction_times) + sum(loss_times)):.2f} seconds.")
        print(f"Fitness calculation took {t_fit_end - t_fit_start:.2f} seconds.")

        # average fitness
        t_current_avg_fitness_start = time.time()
        current_average_fitness = average_fitness(fitness_scores, population)
        t_current_avg_fitness_end = time.time()
        print(f"Average fitness calculation took {t_current_avg_fitness_end - t_current_avg_fitness_start:.2f} seconds.")
        average_fitness_history.append(current_average_fitness)

        # max fitness
        t_max_fitness_start = time.time()
        gen_best_fitness, gen_best_individual = max_fitness(fitness_scores, population)
        t_max_fitness_end = time.time()
        print(f"Max fitness calculation took {t_max_fitness_end - t_max_fitness_start:.2f} seconds.")
        max_fitness_history.append(gen_best_fitness)

        print(f"Generation {generation}: Average Fitness = {current_average_fitness:.4f}, Max Fitness = {gen_best_fitness:.4f}")
        print("Best Individual:", gen_best_individual.to_string())

        # early stopping
        if gen_best_fitness >= 0.9:
            print("Optimal solution found!")
            break

        # elitism
        t_elite_start = time.time()
        ELITE_COUNT = int(PARAMETERS["elitism_rate"])
        sorted_pop = get_population_sorted_by_fitness(population, fitness_scores)
        elites = [individual for individual, _ in sorted_pop[:ELITE_COUNT]]
        t_elite_end = time.time()
        print(f"Elitism selection took {t_elite_end - t_elite_start:.2f} seconds.")

        # selection
        t_operations_start = time.time()
        selector = RouletteSelection()
        parents = selector.create_parents_pool(population, fitness_scores)

        # crossover
        children = run_crossover(parents, PARAMETERS["population_size"] - ELITE_COUNT, PARAMETERS["crossover_rate"])

        # mutation
        mutated_children = run_mutations(children, PARAMETERS["max_tree_height"], PARAMETERS["mutation_rate"])

        # next generation
        population = elites + mutated_children
        t_operations_end = time.time()
        print(f"Selection, Crossover, and Mutation took {t_operations_end - t_operations_start:.2f} seconds.")

        t_gen_end = time.time()
        print(f"Generation {generation} took {t_gen_end - t_gen_start:.2f} seconds.\n")

    t_training_end = time.time()
    print(f"Total training time: {t_training_end - t_setup_start:.2f} seconds.")
    print("The process is completed")
    print("Elapsed generations: ", generation)
    print(average_fitness_history)
    plot_average_and_max_fitness_history(average_fitness_history, max_fitness_history)

    # extract best model after evolution
    best_fitness, best_tree = max_fitness(fitness_scores, population)

    # get predictions
    test_preds = get_predictions(best_tree, test_data, variables)
    test_labels = [1 if p > 0.5 else 0 for p in test_preds]

    # compute accuracy
    test_accuracy = sum(a == b for a, b in zip(test_labels, test_targets)) / len(test_targets)

    print("Test accuracy:", test_accuracy)
    print("Best Tree Expression:", best_tree.to_string())
