import pandas as pd
import numpy as np
import logging
import time
import os

def find_team_and_player_data(df, team_datapath, player_datapath):
    """Find and map team and player data in the given DataFrame."""
    logging.info("Finding unique team IDs...")
    unique_team_ids = df['team_id'].unique()
    logging.info("Unique team IDs found: %s", unique_team_ids)

    logging.info("Loading team data...")
    team_df = pd.read_csv(team_datapath)
    team_id_map = team_df.set_index('NBA_Current_Link_ID')['Current_BBRef_Team_Name'].to_dict()
    df['team_name'] = df['team_id'].map(team_id_map)
    df.loc[df['team_id'] == -1, 'team_name'] = 'Ball'
    
    logging.info("Dropping rows with missing team names...")
    df.dropna(subset=['team_name'], inplace=True)

    logging.info("Finding unique player IDs...")
    unique_player_ids = df['player_id'].unique()
    logging.info("Unique player IDs found: %s", unique_player_ids)

    logging.info("Loading player data...")
    player_df = pd.read_csv(player_datapath, encoding='ISO-8859-1')
    player_df = player_df.drop_duplicates(subset='NBAID')
    player_id_map = player_df.set_index('NBAID')['NBAName'].to_dict()
    df['player_name'] = df['player_id'].map(player_id_map)
    df.loc[df['player_id'] == -1, 'player_name'] = 'Ball'

    logging.info("Dropping rows with missing player names...")
    df.dropna(subset=['player_name'], inplace=True)

    logging.info("Successfully found team and player data.")
    return df

def find_game_start(df):
    """Find the index where the game starts and slice df to start here."""
    logging.info("Finding first player ID...")
    valid_player_ids = df[df['player_id'] != -1]['player_id'].unique()
    first_player_id = valid_player_ids[0] if valid_player_ids.size > 0 else None
    logging.info("First valid player ID: %s", first_player_id)

    logging.info("Determining start of game...")
    decrease_index = None
    for i in range(1, len(df)):
        if df['shot_clock'].iloc[i] < df['shot_clock'].iloc[i - 1]:
            decrease_index = i
            logging.info("Decrease found at index: %d, shot_clock value: %s", decrease_index, df['shot_clock'].iloc[i])
            break

    if decrease_index is not None:
        logging.info("Slicing DataFrame from index: %d", decrease_index)
        df = df.iloc[decrease_index:]

        if first_player_id is not None:
            player_index = df[df['player_id'] == first_player_id].index.min()
            if player_index is not None:
                df = df.iloc[player_index:]

    logging.info("Successfully found game start.")
    return df

def downsample_data(df, multiple): 
    """Downsamples data by a given multiple."""
    logging.info("Downsampling...")
    set_size = 11  # Reading is taken for 10 players + ball every 0.04 seconds by default
    downsampled_rows = []

    for i in range(0, len(df), set_size * multiple):
        downsampled_rows.append(df.iloc[i:i + set_size])

    df = pd.concat(downsampled_rows).reset_index(drop=True)
    
    logging.info("Dropping null values...")
    df.dropna(inplace=True)

    logging.info("Successfully downsampled data by factor of %d.", multiple)
    return df

def calculate_trajectories(df):
    """Calculates velocity and angle of each player at each point."""
    logging.info("Calculating trajectories...")

    if len(df) >= 11:
        df['prev_x_loc'] = df['x_loc'].shift(11)
        df['prev_y_loc'] = df['y_loc'].shift(11)
    df.loc[:9, 'prev_x_loc'] = df.loc[:10, 'x_loc']
    df.loc[:9, 'prev_y_loc'] = df.loc[:10, 'y_loc']

    df['delta_x'] = df['x_loc'] - df['prev_x_loc']
    df['delta_y'] = df['y_loc'] - df['prev_y_loc']
    df['velocity'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    df['angle'] = np.arctan2(df['delta_y'], df['delta_x'])

    logging.info("Successfully calculated trajectory data.")
    return df

def create_stats(datapath, team_datapath, player_datapath, downsample_amount):
    """Combines above functions to return a dataframe with all required data for analysis."""

    logging.info("Starting data processing")
        
    logging.info("Step 1 - Reading data from datapath")
    start_time = time.time()
    logging.info("Reading data...")
    df = pd.read_csv(datapath)
    elapsed_time = time.time() - start_time
    logging.info("Time taken to read data from datapath: %.2f seconds", elapsed_time)
    
    logging.info("Step 2 - Filter out radius")
    start_time = time.time()
    logging.info("Dropping unnecessary columns...")
    df.drop(columns=['radius'], inplace=True, errors='ignore')
    elapsed_time = time.time() - start_time
    logging.info("Time taken to drop radius column: %.2f seconds", elapsed_time)

    logging.info("Step 3 - Fetching and including team and player names")
    start_time = time.time()
    df = find_team_and_player_data(df, team_datapath, player_datapath)
    elapsed_time = time.time() - start_time
    logging.info("Time taken to fetch and include team and player data: %.2f seconds", elapsed_time)

    logging.info("Step 4 - Find start of game and slice database")
    start_time = time.time()
    df = find_game_start(df)
    elapsed_time = time.time() - start_time
    logging.info("Time taken to find start of game and slice database: %.2f seconds", elapsed_time)

    logging.info("Step 5 - Downsampling data")
    start_time = time.time()
    df = downsample_data(df, downsample_amount)
    elapsed_time = time.time() - start_time
    logging.info("Time taken to downsample data: %.2f seconds", elapsed_time)

    logging.info("Step 6 - Calculate trajectories")
    start_time = time.time()
    df = calculate_trajectories(df)
    elapsed_time = time.time() - start_time
    logging.info("Time taken to calculate: %.2f seconds", elapsed_time)

    df.dropna(inplace=True)

    logging.info("Finished data processing")
    return df

def process_multiple_csvs(folder_path, n, team_datapath, player_datapath, downsample_amount):
    """Processes the first n CSV files in the given folder and concatenates them into one DataFrame."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    files.sort()  # Sort the files alphabetically, modify as needed for different sorting criteria
    
    all_data = []  # List to store all processed DataFrames
    
    for i, file in enumerate(files[:n]):
        logging.info("Processing file %d: %s", i + 1, file)
        datapath = os.path.join(folder_path, file)
        df = create_stats(datapath, team_datapath, player_datapath, downsample_amount)

        # Append the processed DataFrame to the list
        all_data.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    concatenated_df = pd.concat(all_data, ignore_index=True)

    return concatenated_df

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example Use: Process the first 5 CSV files
    folder_path = "DATA/nba-movement-data-master/datacsv"
    team_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Team_IDs.csv"  
    player_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Player_Ids.csv"
    downsample_amount = 2
    n_files = 5  # Process the first 5 files

    df = process_multiple_csvs(folder_path, n_files, team_datapath, player_datapath, downsample_amount)
    print(df.info())