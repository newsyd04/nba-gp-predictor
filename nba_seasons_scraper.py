from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv3
import pandas as pd
import time
from datetime import date
from tqdm import tqdm
import os

NUM_SEASONS = 5
INCLUDE_CURRENT_SEASON = True
API_SLEEP_SECONDS = 1.5
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def get_recent_seasons(num_seasons=3, include_current=True) -> list[str]:
    """
    Returns a list of recent NBA season strings in "YYYY-YY" format, e.g., "2024-25".
    NBA season is referenced by the starting year, 2024-2025 -> "2024-25".
    Determine current season start year: if month >= 10 (Oct) it is in the season starting this calendar year.
    """
    today = date.today()
    year = today.year
    if today.month < 10:
        season_start = year - 1
    else:
        season_start = year
    seasons = []
    for i in range(num_seasons):
        s = season_start - (0 if include_current else 1) - i
        seasons.append(f"{s}-{str(s+1)[-2:]}")
    return seasons

def fetch_games_for_seasons(season_list: list[str]) -> pd.DataFrame:
    """
    Returns a dataframe of unique games for the given seasons.
    """
    season_frames = []
    for season in season_list:
        print(f"Fetching regular season games for {season} ...")
        gf = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable='Regular Season')
        df = gf.get_data_frames()[0]
        unique_games = df[['GAME_ID','GAME_DATE','TEAM_ID','TEAM_ABBREVIATION','MATCHUP','WL']].drop_duplicates(subset='GAME_ID')
        # adds a season column to the dataframe with the current season
        unique_games['SEASON'] = season
        unique_games.to_csv(os.path.join(OUT_DIR, 'seasons', f"games_{season}.csv"), index=False)
        season_frames.append(unique_games)
        time.sleep(API_SLEEP_SECONDS)
    return pd.concat(season_frames, ignore_index=True)

def fetch_boxscore_advanced_for_game(game_id: str, try_attempts=2) -> list[pd.DataFrame]:
    """
    Returns the advanced boxscore for a single game as a list of dataframes.
    On failure after retries, raises the exception, which is logged with the game id to "failed_games.log".
    """
    for attempt in range(1, try_attempts+1):
        try:
            endpoint = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
            dfs = endpoint.get_data_frames()
            if not dfs:
                raise ValueError(f"No dataframes returned for game {game_id}")

            time.sleep(API_SLEEP_SECONDS)
            return dfs
        except Exception as e:
            print(f"Attempt {attempt} failed for game {game_id}: {e}")
            if attempt == try_attempts:
                if not os.path.exists("failed_games.log"):
                    with open("failed_games.log", "w") as f:
                        f.write(f"{game_id}\n")
                else:
                    with open("failed_games.log", "a") as f:
                        f.write(f"{game_id}\n")
                raise

            time.sleep(API_SLEEP_SECONDS * attempt)  # exponential backoff

def load_processed_game_ids(season_out_dir: str, season_name: str) -> set[str]:
    """
    Returns a set of already processed game IDs from the existing season CSV files.
    Scans files like "{season_name}_table*.csv" in the directory for `gameId` strings already present in those files.
    """
    processed = set()
    if not os.path.exists(season_out_dir):
        return processed

    for file_name in os.listdir(season_out_dir):
        if not file_name.startswith(season_name) or not file_name.endswith('.csv'):
            continue
        file_path = os.path.join(season_out_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                header = file.readline().strip()
                if not header:
                    continue
                columns = [column.strip() for column in header.split(',')]
                # find index of game id column
                game_id_index = None
                for i, column in enumerate(columns):
                    if column.lower() in ('gameid', 'game_id'):
                        game_id_index = i
                        break
                if game_id_index is None:
                    game_id_index = 0

                # iterate remaining lines and add their game id values to processed set
                for line in file:
                    if not line.strip():
                        continue
                    parts = line.rstrip('\n').split(',')
                    if len(parts) <= game_id_index:
                        continue
                    game_id = parts[game_id_index].strip()
                    if game_id:
                        processed.add(game_id)
        except Exception:
            continue

    return processed

def main() -> None:
    already_completed = True
    complete_seasons_dir = os.path.join(OUT_DIR, 'seasons')
    seasons = get_recent_seasons(NUM_SEASONS, include_current=INCLUDE_CURRENT_SEASON)
    not_completed_seasons = []
    for season in seasons:
        if not os.path.exists(os.path.join(complete_seasons_dir, f"games_{season}.csv")):
            already_completed = False
            not_completed_seasons.append(season)
    if not already_completed:
        games_df = fetch_games_for_seasons(not_completed_seasons)
        print(f"Found {len(games_df['GAME_ID'].unique())} unique games for those seasons.")
    else:
        print(f"Seasons already completed.")
        print("Loading existing games list...")
        for season in seasons:
            season_df = pd.read_csv(os.path.join(complete_seasons_dir, f"games_{season}.csv"))
            if season == seasons[0]:
                games_df = season_df
            else:
                games_df = pd.concat([games_df, season_df], ignore_index=True)

    seen = set()
    for season in seasons:
        season_games = games_df[games_df['SEASON'] == season]
        season_out_dir = os.path.join(OUT_DIR, season)
        processed_ids = load_processed_game_ids(season_out_dir, season)
        if processed_ids:
            print(f"Resuming {season}: found {len(processed_ids)} previously processed game ids -> skipping them.")
        seen.update(processed_ids)

        # prepare list of game IDs that haven't been processed yet (respect global seen)
        unique_ids = [game_id for game_id in season_games['GAME_ID'].unique() if "00" + str(game_id) not in seen]
        if not unique_ids:
            print(f"All games for {season} already processed -> skipping.")
            continue

        for game_id in tqdm(unique_ids, desc=f"Games {season}", total=len(unique_ids)):
            str_game_id = "00" + str(game_id)
            season_id = str_game_id[3:5]
            if str_game_id in seen:
                continue
            seen.add(str_game_id)
            try:
                dfs = fetch_boxscore_advanced_for_game(str_game_id)
            except Exception as e:
                print(f"Failed to fetch advanced for {str_game_id}: {e}")
                continue

            season_name = f"{2000 + int(season_id)}-{str(2001 + int(season_id))[-2:]}"
            season_out_dir = os.path.join(OUT_DIR, season_name)
            if not os.path.exists(season_out_dir):
                os.makedirs(season_out_dir, exist_ok=True)

            # dfs is a list of dataframes, common layout -> 1st: players advanced stats, 2nd: teams advanced stats.
            # append each dataframe to a CSV, named by season name and table index, e.g., "2025-26_table0.csv", "2025-26_table1.csv".
            for i, df in enumerate(dfs):
                file_name = f"{season_name}_table{i}.csv"
                if df.empty:
                    continue
                if isinstance(df, pd.DataFrame):
                    if os.path.exists(os.path.join(season_out_dir, file_name)):
                        df.to_csv(os.path.join(season_out_dir, file_name), mode='a', header=False, index=False)
                    else:
                        df.to_csv(os.path.join(season_out_dir, file_name), index=False)

    print("Done. CSVs written to", OUT_DIR)

if __name__ == "__main__":
    main()
