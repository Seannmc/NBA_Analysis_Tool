import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, Lambda
from tensorflow.keras import backend as K
from database_create_stats import create_stats
from log_functions import clear_logging, setup_logging, reset_logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_sequences(data, target, time_steps=10, H=10):
    """
    Creates input-output sequences for the LSTM model.

    Parameters:
    - data: 2D numpy array containing feature data.
    - target: 2D numpy array containing target data (future x and y locations).
    - time_steps: The number of time steps to include in each input sequence.
    - H: Number of future steps for each output sequence (trajectory length).

    Returns:
    - X_seq: 3D numpy array of input sequences.
    - y_seq: 3D numpy array of output sequences, each with a flattened (2 * H) structure.
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - time_steps - H):
        X_seq.append(data[i:i + time_steps])
        
        # Get the next H steps and flatten for trajectory targets (2 * H)
        y_future = target[i + time_steps:i + time_steps + H].flatten()
        if y_future.shape[0] == 2 * H:  # Ensure we have exactly H steps of (x, y) coordinates
            y_seq.append(y_future)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    logging.info(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
    
    return X_seq, y_seq

def train_multimodal_lstm_model(df, time_steps=5, epochs=5, batch_size=16, H=10, M=3):
    """
    Trains a multi-modal LSTM model to predict multiple potential trajectories and their probabilities.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - time_steps: The number of time steps the model should look at when predicting.
    - epochs: Number of training epochs.
    - batch_size: The batch size used during training.
    - H: Number of time steps in each predicted trajectory.
    - M: Number of modes (i.e., possible trajectories).

    Returns:
    - model: The trained multi-modal LSTM model.
    - history: Training history (loss and validation loss).
    """
    logging.info("Starting multi-modal LSTM model training process...")

    required_columns = ['prev_x_loc', 'prev_y_loc', 'delta_x', 'delta_y', 'velocity', 'angle', 'game_clock', 'shot_clock']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"DataFrame is missing required columns. Required: {required_columns}")
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    logging.info("DataFrame contains all required columns.")

    # Define target columns (next x and y location)
    df['next_x_loc'] = df['prev_x_loc'] + df['delta_x']
    df['next_y_loc'] = df['prev_y_loc'] + df['delta_y']
    
    X = df[['prev_x_loc', 'prev_y_loc', 'delta_x', 'delta_y', 'velocity', 'angle', 'game_clock', 'shot_clock']].values
    y = df[['next_x_loc', 'next_y_loc']].values

    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps, H)
    logging.info(f"Data prepared. Shape of X_seq: {X_seq.shape}, Shape of y_seq: {y_seq.shape}")

    # Define the multi-modal LSTM model
    logging.info("Defining multi-modal LSTM model...")
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))

    # Dense layer to produce multiple trajectories (2H outputs for each mode) and probabilities (1 output per mode)
    model.add(Dense((2 * H) * M + M))  # M modes with 2H coordinates and 1 probability per mode

    # Separate trajectories and apply softmax to mode probabilities
    def split_outputs(x):
        trajectories = x[:, :2 * H * M]
        probabilities = x[:, 2 * H * M:]
        probabilities = K.softmax(probabilities, axis=-1)
        return K.concatenate([K.reshape(trajectories, (-1, M, 2 * H)), K.expand_dims(probabilities, axis=-1)], axis=-1)

    model.add(Lambda(split_outputs))

    # Compile the model with custom loss
    model.compile(optimizer='adam', loss=multi_modal_loss(M, H))
    logging.info("Model compiled successfully.")

    # Model summary
    model.summary()

    # Train the model
    logging.info(f"Training the model for {epochs} epochs...")
    history = model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Plot loss over epochs
    logging.info("Plotting training and validation loss...")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    test_loss = model.evaluate(X_seq, y_seq)
    logging.info(f'Test Loss: {test_loss}')

    return model, history

def multi_modal_loss(M, H):
    """
    Custom loss function to handle multi-modal predictions. Computes the weighted loss across multiple modes.

    Parameters:
    - M: Number of modes.
    - H: Number of time steps in each predicted trajectory.

    Returns:
    - loss: The calculated multi-modal loss.
    """
    def loss(y_true, y_pred):
        mode_probabilities = y_pred[:, :, -1]  # Last value in each mode is the probability
        trajectory_predictions = y_pred[:, :, :-1]  # The first 2H values are the trajectory coordinates
        
        y_true_reshaped = K.reshape(y_true, (-1, 1, 2 * H))  # Reshape y_true to match y_pred's shape

        # Calculate squared differences
        squared_diff = K.square(trajectory_predictions - y_true_reshaped)

        # Sum squared differences over trajectory points
        trajectory_losses = K.sum(squared_diff, axis=-1)

        # Weight each mode's trajectory loss by its probability
        weighted_losses = mode_probabilities * trajectory_losses

        # Minimize weighted loss across modes
        min_loss = K.min(weighted_losses, axis=-1)

        # Return the average of minimum losses over the batch
        return K.mean(min_loss)

    return loss


def evaluate_model(model, X_seq, y_seq, M, H, scaler_y):
    """
    Evaluate the model's performance using additional error metrics.
    
    Parameters:
    - model: Trained LSTM model.
    - X_seq: Input sequences for evaluation.
    - y_seq: True output sequences (ground truth).
    - M: Number of modes (trajectories).
    - H: Number of time steps in each predicted trajectory.
    - scaler_y: Scaler used for target data (to inverse transform the predictions).
    
    Returns:
    - None (Prints out the error metrics).
    """
    # Get model predictions
    y_pred = model.predict(X_seq)

    # Reshape predictions to extract individual modes and trajectories
    predicted_trajectories = y_pred[:, :, :-1].reshape(-1, M, H, 2)  # Shape: (batch_size, M, H, 2)
    predicted_probabilities = y_pred[:, :, -1]  # Shape: (batch_size, M)
    
    # Inverse transform the predicted and true values to get them back in the original scale
    y_true = scaler_y.inverse_transform(y_seq)
    predicted_trajectories = scaler_y.inverse_transform(predicted_trajectories.reshape(-1, 2))

    # Reshape to get back the correct format
    predicted_trajectories = predicted_trajectories.reshape(-1, M, H, 2)
    
    # Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
    mae = mean_absolute_error(y_true.reshape(-1, 2 * H), predicted_trajectories.reshape(-1, 2 * H))
    mse = mean_squared_error(y_true.reshape(-1, 2 * H), predicted_trajectories.reshape(-1, 2 * H))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true.reshape(-1, 2 * H), predicted_trajectories.reshape(-1, 2 * H))
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

    # Mode Prediction Accuracy
    # For simplicity, assume we take the mode with the highest probability
    mode_predicted = np.argmax(predicted_probabilities, axis=1)
    mode_true = np.argmin(np.linalg.norm(y_true.reshape(-1, 2, H) - predicted_trajectories, axis=-1), axis=1)
    
    mode_accuracy = np.mean(mode_predicted == mode_true)
    print(f"Mode Prediction Accuracy: {mode_accuracy * 100:.2f}%")

# Example usage:
if __name__ == '__main__':
    datapath = "DATA/nba-movement-data-master/datacsv/0021500001.csv"
    team_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Team_IDs.csv"
    player_datapath = "DATA/nba-movement-data-master/datacsv/NBA_Player_Ids.csv"
    downsample_amount = 2
    n_files = 5

    log_file = os.path.join("logs", "Model_training.log")
    clear_logging(log_file)
    setup_logging(log_file)

    df = create_stats(datapath, team_datapath, player_datapath, 3)
    model, history = train_multimodal_lstm_model(df, H=10, M=3)

    # Now evaluate the model
    X = df[['prev_x_loc', 'prev_y_loc', 'delta_x', 'delta_y', 'velocity', 'angle', 'game_clock', 'shot_clock']].values
    y = df[['next_x_loc', 'next_y_loc']].values
    
    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps=5, H=10)

    # Evaluate the model using the defined metrics
    evaluate_model(model, X_seq, y_seq, M=3, H=10, scaler_y=scaler_y)
    reset_logging()