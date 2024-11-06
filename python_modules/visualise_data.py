import seaborn as sns
import os
import logging
import matplotlib.pyplot as plt
from python_modules.plotting_court import plot_court
from python_modules.database_analyse_team import gather_team_full_possessions_data, gather_team_offensive_possessions_data
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  

def create_heatmap(player_name, df, team_id):
    logging.info(f"Starting heatmap creation for {player_name}")


    fig, ax = plt.subplots(figsize=(15, 8))
    plot_court(ax)

    player_data = df[(df['player_name'] == player_name) & (df['team_id'] == team_id)]

    sns.kdeplot(
        data=player_data, x='x_loc', y='y_loc', fill=True, thresh=0, levels=50,
        cmap="viridis", ax=ax, zorder=-1, alpha=0.8, bw_adjust=1.5
    )
    ax.set_title(f"Heatmap for {player_name}", fontsize=16)
    ax.axis('off')  

    heatmap_path = os.path.join("heatmaps", f"heatmap_{player_name}.png")
    plt.savefig(heatmap_path, dpi=80)
    plt.close(fig)

    logging.info(f"Finished heatmap creation for {player_name}")
    return heatmap_path

def plot_heatmaps(df, team_id):
    logging.info("Starting heatmap plotting...")


    unique_player_names = [name for name in df['player_name'].unique() if "ball" not in name.lower()]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(create_heatmap, player_name, df, team_id): player_name for player_name in unique_player_names}

        for future in as_completed(futures):
            player_name = futures[future]
            try:
                result = future.result()
                logging.info(f"Heatmap saved at {result} for {player_name}")
            except Exception as e:
                logging.error(f"Error generating heatmap for {player_name}: {e}")

def plot_offensive_trajectories(df, team_id, possession_id):
    logging.info("Filtering DF for specified team id and possession...")

    filtered_df = df[
        ((df['team_id'] == team_id) | 
        (df['player_id'] == -1)) & 
        (df['event_id'] == possession_id)
    ]

    logging.info("Gathering unique player names...")
    unique_player_names = filtered_df['player_name'].unique()

    logging.info("Fetching team name...")
    team_name = filtered_df['team_name'].unique()[0] if not filtered_df['team_name'].isnull().all() else "Unknown Team"

    logging.info("Setting up plot...")
    os.makedirs("Movement_graphs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 8))
    plot_court(ax)  

    for player_name in tqdm(unique_player_names, desc="Plotting player positions", unit="player"):
        logging.info(f"Plotting player {player_name} ...")
        player_data = filtered_df[filtered_df['player_name'] == player_name]
        
        if player_name.lower() == "ball":
            ax.scatter(player_data['x_loc'], player_data['y_loc'], marker='o', s=30, edgecolor='black', facecolor='none', label='Ball')
        else:
            ax.scatter(player_data['x_loc'], player_data['y_loc'], marker='o', s=10, label=player_name)

    plt.title(f"Player Positions During Offensive Possession {possession_id} for the {team_name}")
    plt.xlabel('Court X Location')
    plt.ylabel('Court Y Location')


    ax.legend(loc='upper right', fontsize='small')


    filename = os.path.join("Movement_graphs", f"Offensive possession_{possession_id}_team_{team_id}.png")
    plt.savefig(filename, dpi=80)
    plt.close(fig)

    logging.info(f"Figure saved as {filename}")

def plot_full_trajectories(df, team_id, possession_id):
    logging.info("Filtering DF for specified team id and possession...")


    filtered_df = df[
        ((df['team_id'] == team_id) | 
        (df['player_id'] == -1)) & 
        (df['event_id'] == possession_id)
    ]

    logging.info("Gathering unique player names...")
    unique_player_names = filtered_df['player_name'].unique()

    logging.info("Fetching team name...")
    team_name = filtered_df['team_name'].unique()[0] if not filtered_df['team_name'].isnull().all() else "Unknown Team"

    logging.info("Setting up plot...")
    os.makedirs("Movement_graphs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 8))
    plot_court(ax)

    for player_name in tqdm(unique_player_names, desc="Plotting player positions", unit="player"):
        logging.info(f"Plotting player {player_name} ...")
        player_data = filtered_df[filtered_df['player_name'] == player_name]
        
        if player_name.lower() == "ball":
            ax.scatter(player_data['x_loc'], player_data['y_loc'], marker='o', s=30, edgecolor='black', facecolor='none', label='Ball')
        else:
            ax.scatter(player_data['x_loc'], player_data['y_loc'], marker='o', s=10, label=player_name)

    plt.title(f"Player Positions During Possession {possession_id} for the {team_name}")
    plt.xlabel('Court X Location')
    plt.ylabel('Court Y Location')


    ax.legend(loc='upper right', fontsize='small')

    filename = os.path.join("Movement_graphs", f"Full possession_{possession_id}_team_{team_id}.png")
    plt.savefig(filename, dpi=80)
    plt.close(fig)

    logging.info(f"Figure saved as {filename}")



def plot_correlation_matrix(df):
    """
    Plots a colored correlation matrix for all features in the DataFrame.

    Parameters:
    - df: The pandas DataFrame containing the data.

    Returns:
    - None (Displays the correlation matrix plot)
    """
    
    df = df.select_dtypes(include=['number'])
    

    corr_matrix = df.corr()
    

    plt.figure(figsize=(10, 8)) 
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1,
                linewidths=0.5, square=True, cbar_kws={'shrink': 0.75})
    

    plt.title('Correlation Matrix of Features', fontsize=16)
    plt.tight_layout()
    

    plt.show()


if __name__ == "__main__":
    team_id = 1610612765
    datapath = "DATA/nba-movement-data-master/datacsv/0021500001.csv"
    team_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Team_IDs.csv"  
    player_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Player_Ids.csv"
    downsample_amount = 2
    possession_id = 36
    log_file = os.path.join("logs", "Trajectories.log")


    logging.basicConfig(filename=log_file, level=logging.INFO)
