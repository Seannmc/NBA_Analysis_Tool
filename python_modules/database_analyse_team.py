import logging
import time
from python_modules.database_create_stats import create_stats

def gather_team_offensive_possessions_data(datapath, team_datapath, player_datapath, teamid, downsample_amount):
    start_time = time.time() 

    logging.info("Creating stats...")
    df = create_stats(datapath, team_datapath, player_datapath, downsample_amount)  

    # Filter the DataFrame for the specified team
    logging.info("Filtering DataFrame for the specified team...")
    team_df = df[(df['team_id'] == teamid) | (df['player_id'] == -1)].copy()  

    # Reset index after slicing
    team_df.reset_index(drop=True, inplace=True)

    # Create a column for offensive possession  
    team_df['offensive_possession'] = True

    # Iterate through the DataFrame in chunks of 6
    logging.info("Determining offensive possession...")
    for i in range(0, len(team_df), 5):
        # Get the current group of players
        current_group = team_df.iloc[i:i + 5]

        # Check if all players are in the left half
        if all(current_group['x_loc'] < 47):
            team_df.loc[i:i + 4, 'offensive_possession'] = True  # Case for all in left half
        else:
            team_df.loc[i:i + 4, 'offensive_possession'] = False  # Case for at least one out of bounds

    # Omly keeping offensive possessions
    team_df = team_df[team_df['offensive_possession']]

    elapsed_time = time.time() - start_time  
    logging.info("Time taken to gather offensive positions: %.2f seconds", elapsed_time)

    logging.info("Finished gathering offensive positions data")
    return team_df


def gather_team_full_possessions_data(datapath, team_datapath, player_datapath, teamid, downsample_amount):
    
    start_time = time.time()  

    # Creating stats
    logging.info("Creating stats...")
    df = create_stats(datapath, team_datapath, player_datapath, downsample_amount) 
    

    # Filter the DataFrame for the specified team
    logging.info("Filtering DataFrame for the specified team...")
    team_df = df[(df['team_id'] == teamid) | (df['player_id'] == -1)].copy()  

    # Reset index after slicing
    team_df.reset_index(drop=True, inplace=True)
    
    elapsed_time = time.time() - start_time  
    logging.info("Time taken to gather all possessions: %.2f seconds", elapsed_time)

    logging.info("Finished gathering full possessions data")
    return team_df

if __name__ == "__main__":
    team_id = 1610612765
    datapath = "DATA/nba-movement-data-master/datacsv/0021500001.csv"
    team_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Team_IDs.csv"  
    player_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Player_Ids.csv"
    downsample_amount = 2

    df = gather_team_offensive_possessions_data(datapath, team_datapath, player_datapath, team_id, downsample_amount)
    logging.info("Sample of gathered offensive possessions data:\n%s", df.head(21))
