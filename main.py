import pandas as pd
import time
from config import PARAMETERS, TABLE_PATH
from data_loader import load_nba_data
from population import make_population
from fitness import fitness, average_fitness, max_fitness, get_predictions
from selection.roulette import RouletteSelection
from evolve.crossover import run_crossover
from evolve.mutation import run_mutations

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
        t_fit_start = time.time()
        fitness_scores = [fitness(individual, train_target_outputs, train_data, variables) for individual in population]
        t_fit_end = time.time()
        print(f"Fitness calculation took {t_fit_end - t_fit_start:.2f} seconds.")
        t_current_avg_fitness_start = time.time()
        current_average_fitness = average_fitness(fitness_scores, population)
        t_current_avg_fitness_end = time.time()
        print(f"Average fitness calculation took {t_current_avg_fitness_end - t_current_avg_fitness_start:.2f} seconds.")
        average_fitness_history.append(current_average_fitness)
        t_max_fitness_start = time.time()
        best_fitness, best_individual = max_fitness(fitness_scores, population)
        t_max_fitness_end = time.time()
        print(f"Max fitness calculation took {t_max_fitness_end - t_max_fitness_start:.2f} seconds.")
        max_fitness_history.append(best_fitness)
        print(f"Average Fitness = {current_average_fitness:.4f}, Max Fitness = {best_fitness:.4f}")
        print("Best Individual:", best_individual.to_string())

        if best_fitness >= 0.9: # fitness threshold
             print("Optimal solution found!")
             break

        t_operations_start = time.time()
        # selection
        selector = RouletteSelection()
        parents = selector.create_parents_pool(population, fitness_scores)
        # crossover
        children = run_crossover(parents, PARAMETERS["population_size"], PARAMETERS["crossover_rate"])
        # mutations
        mutated_children = run_mutations(children, PARAMETERS["max_tree_height"], PARAMETERS["mutation_rate"])
        population = mutated_children
        t_operations_end = time.time()
        print(f"Selection, Crossover, and Mutation took {t_operations_end - t_operations_start:.2f} seconds.")
        t_gen_end = time.time()
        print(f"Generation {generation} took {t_gen_end - t_gen_start:.2f} seconds.\n")

    print("The process is completed")
    print("Elapsed generations: ", generation)
    print(average_fitness_history)
    
    
    
    # extract best model after evolution
    best_fitness, best_tree = max_fitness(population, train_target_outputs, train_data, variables)

    # get test labels
    test_targets = [row[-1] for row in test_data]  # before you removed target

    # get predictions
    test_preds = get_predictions(best_tree, test_data, variables)
    test_labels = [1 if p > 0.5 else 0 for p in test_preds]

    # compute accuracy
    test_accuracy = sum(a == b for a, b in zip(test_labels, test_targets)) / len(test_targets)

    print("Test accuracy:", test_accuracy)
    print("Best Tree Expression:", best_tree.to_string())
    # plot_average_and_max_fitness_history(average_fitness_history, max_fitness_history)
    # best_individuals_predictions = get_predictions(best_individual, data, variables)
    # plot_target_vs_final_prediction(target_outputs, best_individuals_predictions)
    # plot_target_vs_final_prediction_2d(target_outputs, best_individuals_predictions)
    # best_individual_func = best_individual.to_string()
    # plot_target_vs_final_prediction_functions(target_expression, best_individual_func)

# need to find all files with form of data/XXXX-XX/XXXX-XX_table1.csv

# print("Feature Variables:", variables)
# print("Number of Features:", len(variables))

# print("Train length:", len(train_data))
# print("Test length:", len(test_data))

# print(f"First train row ({len(train_data[0])} columns):", train_data[0])
# print(f"First five train row targets:", [train_data[i][-1] for i in range(5)])
# print(f"First test row ({len(test_data[0])} columns):", test_data[0])

# print("Train target mean:", sum(row[-1] for row in train_data) / len(train_data))
# print("Test target mean:", sum(row[-1] for row in test_data) / len(test_data))

# train_data = pd.DataFrame(train_data, columns=variables + ["target"]) 
# test_data = pd.DataFrame(test_data, columns=variables + ["target"])
# print(train_data.head())
# print(test_data.head())

