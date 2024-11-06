import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from ml_model import train_lstm_model
from log_functions import clear_logging, reset_logging, setup_logging
from database_create_stats import create_stats



def fine_tune_time_steps(df, epochs=5, batch_size=16, time_steps_range=[5, 10, 15, 20]):
    best_time_steps = None
    best_mse = float('inf')
    best_model = None
    best_history = None

    for time_steps in time_steps_range:
        logging.info(f"Training model with {time_steps} time steps...")


        model, history, X_seq, y_seq = train_lstm_model(df, time_steps=time_steps, epochs=epochs, batch_size=batch_size)


        test_loss = model.evaluate(X_seq, y_seq)
        logging.info(f"Model with {time_steps} time steps - Test Loss: {test_loss}")


        if test_loss < best_mse:
            best_mse = test_loss
            best_time_steps = time_steps
            best_model = model
            best_history = history

    logging.info(f"Best model found with {best_time_steps} time steps.")
    

    plt.plot(best_history.history['loss'], label='Train Loss')
    plt.plot(best_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Training and Validation Loss (Best Time Steps: {best_time_steps})")
    plt.show()
    
    return best_model, best_history, best_time_steps, best_mse



if __name__ == '__main__':
    folder_path = "DATA/nba-movement-data-master/datacsv"  
    team_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Team_IDs.csv"  
    player_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Player_Ids.csv"
    datapath = "DATA/nba-movement-data-master/datacsv/0021500001.csv"
    downsample_amount = 2
    n_files = 5  

    log_file = os.path.join("logs", "Model_training.log")
    clear_logging(log_file)
    setup_logging(log_file)


    df = create_stats(datapath, team_datapath, player_datapath, downsample_amount)

    logging.info("Data processed, now training the model...")


    best_model, best_history, best_time_steps, best_mse = fine_tune_time_steps(df, epochs=10, batch_size=16, time_steps_range=[5, 10, 15, 20])


    print(f"Best Time Steps: {best_time_steps}")
    print(f"Best MSE: {best_mse}")