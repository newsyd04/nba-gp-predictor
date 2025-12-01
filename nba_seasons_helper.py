import os
import pandas as pd
import matplotlib.pyplot as plt
from nba_seasons_scraper import get_recent_seasons, NUM_SEASONS, INCLUDE_CURRENT_SEASON, OUT_DIR

def add_wl_column_from_games(stats_csv: list[str], games_csv: list[str], out_csv: list[str] = None) -> None:
    if len(stats_csv) != len(games_csv):
        raise ValueError("stats_csv and games_csv must have the same length.")
    for stats_csv, games_csv, out_csv in zip(stats_csv, games_csv, out_csv or [None]*len(stats_csv)):
        stats_with_wl = _add_wl_column_from_games(stats_csv, games_csv, out_csv)
        print(f"Difference between original and new: {set(stats_with_wl.columns) - set(pd.read_csv(stats_csv, dtype=str).columns)}")

def _add_wl_column_from_games(stats_csv: str, games_csv: str, out_csv: str = None) -> pd.DataFrame:
    """
    Adds a WL column ('W' | 'L') and 'won' column (1 | 0) to your advanced team stats for a season.
    """
    stats = pd.read_csv(stats_csv, dtype=str)
    games = pd.read_csv(games_csv, dtype=str)

    if stats.empty or games.empty:
        raise ValueError("One of the input CSV files is empty.")

    if "WL"in stats.columns:
        print("Stats CSV already has WL column -> skipping.")
        return stats

    # normalise the columns to be the same
    stats.rename(columns={
        "gameId": "GAME_ID",
        "teamId": "TEAM_ID",
        "teamTricode": "TEAM_ABBREVIATION"
    }, inplace=True)

    # build wl lookup table
    wl_map = {}  # (GAME_ID, TEAM_ABBREVIATION) → "W" | "L"

    for _, row in games.iterrows():
        game_id = row["GAME_ID"]
        wl = row["WL"]
        matchup = row["MATCHUP"]
        team_abbr = row["TEAM_ABBREVIATION"]

        # parsing "NYK vs. CHI" or "PHX @ MIN"
        parts = matchup.replace("vs.", "vs").replace("@", "").split()
        # format is always "TEAM1 vs TEAM2" OR "TEAM1 TEAM2"
        # team abbreviations are the first and last tokens
        team1 = parts[0]
        team2 = parts[-1]

        # determine which side the wl belongs to
        if team_abbr == team1:
            wl_map[(game_id, team1)] = wl
            wl_map[(game_id, team2)] = "L" if wl == "W" else "W"
        else:
            wl_map[(game_id, team2)] = wl
            wl_map[(game_id, team1)] = "L" if wl == "W" else "W"

    def lookup(row):
        key = (row["GAME_ID"], row["TEAM_ABBREVIATION"])
        return wl_map.get(key, None)

    # apply the wl to the stats rows
    stats["WL"] = stats.apply(lookup, axis=1)
    stats["won"] = stats["WL"].map({"W": 1, "L": 0})

    if out_csv:
        stats.to_csv(out_csv, index=False)

    # rename columns back to original
    stats.rename(columns={
        "GAME_ID": "gameId",
        "TEAM_ID": "teamId",
        "TEAM_ABBREVIATION": "teamTricode",
        "WL": "wl"
    }, inplace=True)

    return stats

def build_team_csvs(seasons: list[str], stats_csvs: list[str]) -> None:
    """
    Creates team specific folders with CSVs for past seasons and current season.
    """
    teams_dir = os.path.join(OUT_DIR, "teams")
    os.makedirs(teams_dir, exist_ok=True)

    current_season = seasons[0]
    past_seasons = seasons[1:]

    season_dfs = {
        season: pd.read_csv(csv).assign(season=season)
        for season, csv in zip(seasons, stats_csvs)
        if os.path.exists(csv)
    }

    # combine past seasons into one dataframe
    past_df = pd.concat(
        [season_dfs[s] for s in past_seasons if s in season_dfs],
        ignore_index=True
    )

    current_df = season_dfs.get(current_season, pd.DataFrame())

    # get all team abbreviations
    all_teams = pd.concat([past_df, current_df])["teamTricode"].unique()

    for team in all_teams:
        _build_team_csvs_for_team(team, past_df, current_df, teams_dir)

    print("Teams CSVs completed.")
    print(f"{len(all_teams)} teams processed.")

def _build_team_csvs_for_team(team: str, past_df: pd.DataFrame, current_df: pd.DataFrame, teams_dir: str) -> None:
    """
    Builds CSVs for a specific team in its own folder.
    """
    team_dir = os.path.join(teams_dir, team)
    os.makedirs(team_dir, exist_ok=True)

    # filter past and current
    team_past = past_df[past_df["teamTricode"] == team]
    team_current = current_df[current_df["teamTricode"] == team]

    # paths
    past_path = os.path.join(team_dir, "past_seasons.csv")
    current_path = os.path.join(team_dir, "current_season.csv")

    if not team_past.empty:
        team_past.to_csv(past_path, index=False)

    if not team_current.empty:
        team_current.to_csv(current_path, index=False)

    print(f"Built team folder for {team}: "
            f"{len(team_past)} past rows, {len(team_current)} current rows")

def plot_correlations_with_win(dfs: list[pd.DataFrame], titles: list[str]) -> None:
    if not os.path.exists(f"{OUT_DIR}/correlations"):
        os.makedirs(f"{OUT_DIR}/correlations")
    for df, title in zip(dfs, titles):
        print(f"Plotting correlations for {title}...")
        _plot_correlations_with_win(df, title)

def _plot_correlations_with_win(df: pd.DataFrame, title: str) -> None:
    """
    Plots the pearson correlation for all useful numeric columns with the 'won' column.
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64'])
    base_correlation = numeric_columns.corr()['won']
    # removing useless columns like usagePercentage -> a team always uses 100% of its possessions in a game, and so it is a more useful player stat
    pruned_correlation = base_correlation.drop('won').drop('gameId').drop('teamId').drop('usagePercentage')
    correlation = pruned_correlation.sort_values(key=lambda s: s.abs(), ascending=False)

    plt.figure(figsize=(10, 14))
    correlation.sort_values().plot(kind="barh", color=["skyblue" if val > 0 else "salmon" for val in correlation.sort_values()])
    plt.title("Correlation of Statistics with Winning")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlations/{title}.png")
    # plt.show()

def _print_correlations_with_win(df: pd.DataFrame, title: str) -> None:
    numeric_columns = df.select_dtypes(include=['float64', 'int64'])
    base_correlation = numeric_columns.corr()['won']
    pruned_correlation = base_correlation.drop('won').drop('gameId').drop('teamId').drop('usagePercentage')
    correlation = pruned_correlation.sort_values(key=lambda s: s, ascending=False)
    print(f"Correlation of Statistics with Winning ({title}):")
    print(correlation)

if __name__ == "__main__":
    # run_type = "add_wl"
    # run_type = "build_team_csvs"
    # run_type = "plot_correlation_per_season"
    # run_type = "plot_correlation_combined"
    run_type = "print_correlation_combined"
    seasons = get_recent_seasons(NUM_SEASONS, include_current=INCLUDE_CURRENT_SEASON)
    stats_csvs = [f"{OUT_DIR}/{season}/{season}_table1.csv" for season in seasons]
    if run_type == "add_wl":
        games_csvs = [f"{OUT_DIR}/seasons/games_{season}.csv" for season in seasons]
        out_csvs = [f"{OUT_DIR}/{season}/{season}_table1.csv" for season in seasons]
        add_wl_column_from_games(stats_csvs, games_csvs, out_csvs)
    elif run_type == "build_team_csvs":
        build_team_csvs(seasons, stats_csvs)
    elif run_type == "plot_correlation_per_season":
        plot_correlations_with_win([pd.read_csv(csv) for csv in stats_csvs], seasons)
    elif run_type == "plot_correlation_combined":
        # plots data across all seasons combined into one correlation chart
        combined_df = pd.concat([pd.read_csv(csv) for csv in stats_csvs], ignore_index=True)
        _plot_correlations_with_win(combined_df, "combined_seasons")
    elif run_type == "print_correlation_combined":
        combined_df = pd.concat([pd.read_csv(csv) for csv in stats_csvs], ignore_index=True)
        _print_correlations_with_win(combined_df, "combined_seasons")
    else:
        raise ValueError(f"Unknown run_type: {run_type}")
