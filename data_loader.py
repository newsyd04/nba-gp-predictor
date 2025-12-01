import glob
import pandas as pd
from gp_config import ROLLING_COLUMNS, TRAIN_SEASONS, TRAIN_CURRENT, TEST_SEASON

def load_all_games_data(path: str) -> pd.DataFrame:
    """
    Loads all games data from all seasons' CSV files and returns a time ordered DataFrame. 
    """
    files = glob.glob(path)
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

    # convert to string, then add "00" to the front if not present
    df["gameId"] = df["gameId"].astype(str).str.zfill(10)

    # extract numeric season year (e.g. 0022500243 -> 2025)
    df["seasonNumber"] = df["gameId"].astype(str).str[3:5].astype(int) + 2000

    # sort in time order per team
    df = df.sort_values(["teamId", "seasonNumber", "gameId"]).reset_index(drop=True)

    return df

def add_last_10_features(df: pd.DataFrame, rolling_columns: list[str]) -> pd.DataFrame:
    """
    Adds last 10 rolling mean features.
    """
    df_last_10 = (df.groupby("teamId")[rolling_columns].apply(lambda x: x.shift(1).rolling(10, min_periods=1).mean()))

    df_last_10.columns = [c + "_last10" for c in df_last_10.columns]
    df = pd.concat([df.reset_index(drop=True), df_last_10.reset_index(drop=True)], axis=1)

    return df

def add_3_year_features(df: pd.DataFrame, rolling_columns: list[str]) -> pd.DataFrame:
    """
    Adds 3 season rolling mean features.
    """
    df_3_year = []

    for _, g in df.groupby("teamId"):
        g = g.sort_values(["seasonNumber", "gameId"])

        season_means = (g.groupby("seasonNumber")[rolling_columns].mean().shift(1).rolling(3, min_periods=1).mean())

        g = g.merge(season_means, left_on="seasonNumber", right_index=True, suffixes=("", "_3Year"), how="left")

        df_3_year.append(g)

    df = pd.concat(df_3_year).sort_values(["teamId", "seasonNumber", "gameId"])
    return df

def enforce_min_games(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """
    Removes rows that do not have enough past games to form last 10.
    """
    df["games_played"] = df.groupby(["teamId", "seasonNumber"]).cumcount()
    return df[df["games_played"] >= min_games].reset_index(drop=True)

def split_train_test_by_season(df: pd.DataFrame, train_seasons: list[int], train_current: int, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Makes a train test split.
    """
    train_df = df[df["seasonNumber"].isin(train_seasons + [train_current])]
    test_df  = df[df["seasonNumber"] == test_season]

    return train_df, test_df

def build_matches(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Converts team rows into Team A vs Team B matches rows.
    """
    base = df[["gameId", "teamId", "won"] + feature_columns]
    A = base.copy()
    B = base.copy()

    A.columns = ["gameId"] + [f"A_{column}" if column != "gameId" else column for column in A.columns[1:]]
    B.columns = ["gameId"] + [f"B_{column}" if column != "gameId" else column for column in B.columns[1:]]

    match = A.merge(B, on="gameId")
    # removes self matches
    match = match[match["A_teamId"] != match["B_teamId"]]
    match = match.reset_index(drop=True)

    return match

def build_delta_features(matches: pd.DataFrame, rolling_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Builds delta features for both last 10 and 3 year.
    """
    delta_columns = []

    for column in rolling_columns:
        for suffix in ["_last10", "_3Year"]:
            A_column = f"A_{column}{suffix}"
            B_column = f"B_{column}{suffix}"
            delta_column = f"delta_{column}{suffix}"
            matches[delta_column] = matches[A_column] - matches[B_column]
            delta_columns.append(delta_column)

    return matches, delta_columns

def build_matrix_tables(train_matches: pd.DataFrame, test_matches: pd.DataFrame, delta_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns X and y for training and testing.
    """
    X_train = train_matches[delta_columns]
    y_train = train_matches["A_won"].astype(int)

    X_test = test_matches[delta_columns]
    y_test = test_matches["A_won"].astype(int)

    train = pd.concat([X_train, y_train.rename("target")], axis=1).dropna()
    test  = pd.concat([X_test,  y_test.rename("target")],  axis=1).dropna()

    return train, test

def load_nba_data(seasons_path: str) -> tuple[list[str], list, list]:
    """
    Converts the raw nba data into train and test sets.
    Returns variables, train_data, test_data.
    """

    df = load_all_games_data(seasons_path)
    df = add_last_10_features(df, ROLLING_COLUMNS)
    df = add_3_year_features(df, ROLLING_COLUMNS)
    df = enforce_min_games(df, min_games=10)
    # df.to_csv("load_all_games_data.csv", index=False)
    # df.to_csv("add_last_10_features.csv", index=False)
    # df.to_csv("add_3_year_features.csv", index=False)
    # df.to_csv("enforce_min_games.csv", index=False)

    train_df, test_df = split_train_test_by_season(df, TRAIN_SEASONS, TRAIN_CURRENT, TEST_SEASON)
    print(f"Train_df shape: {train_df.shape}")
    print(f"Test_df shape: {test_df.shape}")

    FEATURE_COLUMNS = [column for column in df.columns if column.endswith("_last10") or column.endswith("_3Year")]

    train_matches = build_matches(train_df, FEATURE_COLUMNS)
    test_matches = build_matches(test_df, FEATURE_COLUMNS)

    train_matches, delta_columns = build_delta_features(train_matches, ROLLING_COLUMNS)
    test_matches, _ = build_delta_features(test_matches,  ROLLING_COLUMNS)

    train, test = build_matrix_tables(train_matches, test_matches, delta_columns)

    variables = delta_columns
    train_data = train.values.tolist()
    test_data = test.values.tolist()

    return variables, train_data, test_data
