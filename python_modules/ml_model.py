import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from log_functions import clear_logging, reset_logging, setup_logging
from database_create_stats import create_stats


# LSTM Model Creation
def create_lstm_model(seq_length=10, feature_dim=4, lstm_units=128):
    logging.info("Defining LSTM model...")

    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, feature_dim)))
    model.add(LSTM(lstm_units))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2)) 
    model.compile(optimizer=Adam(), loss='mse')

    logging.info("LSTM model defined and compiled.")

    return model


def create_sequences(data, target, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(data) - time_steps):
        X_seq.append(data[i:i + time_steps])
        y_seq.append(target[i + time_steps])
    return np.array(X_seq), np.array(y_seq)



def train_lstm_model(df, time_steps=10, epochs=5, batch_size=16):
    logging.info("Starting LSTM model training process...")


    required_columns = ['prev_x_loc', 'prev_y_loc', 'delta_x', 'delta_y', 'velocity', 'angle', 'game_clock', 'shot_clock']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    logging.info("Preparing the data...")


    df['next_x_loc'] = df['prev_x_loc'] + df['delta_x']
    df['next_y_loc'] = df['prev_y_loc'] + df['delta_y']


    X = df[['prev_x_loc', 'prev_y_loc', 'delta_x', 'delta_y', 'velocity', 'angle', 'game_clock', 'shot_clock']].values
    y = df[['next_x_loc', 'next_y_loc']].values


    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)


    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

    logging.info(f"Data prepared. Shape of X_seq: {X_seq.shape}, Shape of y_seq: {y_seq.shape}")

    logging.info("Defining LSTM model...")
    model = create_lstm_model(seq_length=time_steps, feature_dim=8)


    model.summary()


    logging.info(f"Training the model for {epochs} epochs...")
    history = model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2)


    test_loss = model.evaluate(X_seq, y_seq)
    logging.info(f'Test Loss: {test_loss}')

    return model, history, X_seq, y_seq


def evaluate_model(model, X_test, y_test):
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model evaluation completed. MSE: {mse}")
    return mse



if __name__ == '__main__':
    folder_path = "DATA/nba-movement-data-master/datacsv"  
    team_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Team_IDs.csv"  
    player_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Player_Ids.csv"
    datapath = "DATA/nba-movement-data-master/datacsv/0021500001.csv"
    downsample_amount = 2
    n_files = 5  #

    log_file = os.path.join("logs", "Model_training.log")
    clear_logging(log_file)
    setup_logging(log_file)


    df = create_stats(datapath, team_datapath, player_datapath, downsample_amount)

    logging.info("Data processed, now training the model...")

    model = train_lstm_model(df, time_steps=10, epochs=5, batch_size=16)

    reset_logging()
