import pandas as pd
from data_loader import load_nba_data

# need to find all files with form of data/XXXX-XX/XXXX-XX_table1.csv
path = "data/*/*_table1.csv"
variables, train_data, test_data = load_nba_data(path)

print("Feature Variables:", variables)
print("Number of Features:", len(variables))

print("Train length:", len(train_data))
print("Test length:", len(test_data))

print(f"First train row ({len(train_data[0])} columns):", train_data[0])
print(f"First test row ({len(test_data[0])} columns):", test_data[0])

print("Train target mean:", sum(row[-1] for row in train_data) / len(train_data))
print("Test target mean:", sum(row[-1] for row in test_data) / len(test_data))

train_data = pd.DataFrame(train_data, columns=variables + ["target"]) 
test_data = pd.DataFrame(test_data, columns=variables + ["target"])
print(train_data.head())
print(test_data.head())
