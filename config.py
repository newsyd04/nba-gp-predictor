# ---- GP PARAMETER DEFINITIONS ----
PARAMETERS = {
    "population_size": 150,
    "max_tree_height": 8,
    "max_generations": 100,
    "selection_method": "roulette",
    "crossover_rate": 0.9,
    "mutation_rate": 0.1,
    "elitism_rate": 5,
}

# ---- FEATURE DEFINITIONS ----
ROLLING_COLUMNS = [
    "netRating",
    "PIE",
    "trueShootingPercentage",
    "effectiveFieldGoalPercentage",
    "turnoverRatio",
    "offensiveReboundPercentage",
    "defensiveReboundPercentage",
    "assistRatio"
]

#---- DATA SPLIT DEFINITIONS ----
TRAIN_SEASONS  = [2021, 2022, 2023]
TRAIN_CURRENT = 2024
TEST_SEASON   = 2025

#---- DATA PATHS ----
TABLE_PATH = "data/*/*_table1.csv"

# ---- GP DEFINITIONS ----
OPERATIONS = ['+', '-', '*', '/', 'abs(x)', 'log(|x| + 1)', 'tanh', 'relu']
TERMINALS = ['var', 'const']
