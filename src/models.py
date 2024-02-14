from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import numpy as np
from joblib import dump, load
import torch
import torch.nn as nn
import os
import random

def forecast_and_evaluate(forecaster, y_train, y_test):
    forecaster.fit(y_train)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    return y_pred, forecaster

def forecast_and_evaluate_pytorch(model, y_train, y_test, max_sequence_length=12*2):
    # Ensure model is in evaluation mode
    model.eval()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare the initial sequence from the end of y_train
    if len(y_train) > max_sequence_length:
        initial_sequence = y_train[-max_sequence_length:]
    else:
        initial_sequence = y_train

    # Predictions
    y_pred = []
    current_sequence = np.array([initial_sequence])

    with torch.no_grad():
        for _ in range(len(y_test)):
            seq_tensor = torch.tensor(current_sequence, dtype=torch.float32).to(device)
            output = model(seq_tensor)
            next_value = output.cpu().numpy().flatten()[0]
            y_pred.append(next_value)

            # Update the sequence with the predicted value
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1] = next_value

    return np.array(y_pred)

def initialize_random_seed(seed_value=123):
    """
    Initializes the random seed for NumPy and PyTorch to ensure reproducibility.

    Parameters:
    seed_value (int): The seed value for random number generation. Default is 123.

    Returns:
    None
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # if using CUDA with torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def initialize_models():

    # Set the hyperparameters
    input_size = 1  # Number of features
    hidden_size = 4  # Number of features in hidden state. Not too complex because it will overfit
    num_layers = 1  # Number of stacked layers
    output_size = 1  # Number of output time points
    num_heads = 1  # Number of heads in the Transformer model
    dropout = 0.05 # to prevent overfitting

    # Initialize statistical forecasting models
    models = {
        # NaiveForecaster: Basic model using the last observed value; good for baseline comparison. 'sp' captures annual seasonality.
        'NaiveForecaster': NaiveForecaster(strategy="last", sp=12),

        # ThetaForecaster: Smoothing and transformation, effective for trending and seasonal data. 'sp' indicates seasonality period.
        'ThetaForecaster': ThetaForecaster(sp=12, deseasonalize=True),

        # ExponentialSmoothing: Advanced smoothing for data with trends and multiplicative seasonality. 'sp' for seasonality, 'use_boxcox' for variance stabilization.
        'ExponentialSmoothing': ExponentialSmoothing(
            trend='additive', damped_trend=True, seasonal='additive', sp=12,
            use_boxcox=True, initialization_method='estimated',
            optimized=True, random_state=123
        ),

        # TBATS: Versatile for complex seasonal patterns, multiple seasonality levels, trend and ARMA errors. 'sp' array for multiple seasonal periods.
        'TBATS': TBATS(
            sp=[12], use_trend=True, use_box_cox=True,
            use_damped_trend=True, use_arma_errors=True,
        ),

        # ARIMA: AutoARIMA for non-seasonal data, extensive range of parameters for model flexibility. Includes differencing tests for automatic 'd' selection and information criteria for model optimization.
        'ARIMA': AutoARIMA(
            sp=12,
            start_p=1, start_q=1, max_p=4, max_q=4,
            seasonal=False, suppress_warnings=True,
            test='kpss', information_criterion='aic', n_fits=200,
            out_of_sample_size=2, scoring='mse'
        ),

        # SARIMAX: Seasonal ARIMA, fitting a wide range of seasonal patterns. Uses seasonal differencing tests for 'D' and considers information criteria for model selection.
        'SARIMAX': AutoARIMA(
            sp=12,
            start_p=1, start_q=1, max_p=4, max_q=4,
            seasonal=True, suppress_warnings=True,
            test='kpss', information_criterion='aic', n_fits=200,
            out_of_sample_size=2, scoring='mse'
        ),

        # Prophet: Flexible for daily data, robust to missing data and trend shifts. 'seasonality_mode' for additive/multiplicative seasonality.
        'Prophet': Prophet(
            # growth='linear', # Use 'linear' or 'logistic' to model trend; 'logistic' is useful if there's a known saturation maximum
            yearly_seasonality='auto', # Automatically infer yearly seasonality
            # weekly_seasonality='auto', # Automatically infer weekly seasonality
            # daily_seasonality='auto', # Automatically infer daily seasonality
            # holidays=None, # Include holiday effects if relevant
            seasonality_mode='additive', # Use 'multiplicative' or 'additive'. Multiplicative is often more appropriate when the seasonal effect is proportional to the trend
            # seasonality_prior_scale=10.0, # Larger values allow the model to fit larger seasonal fluctuations, smaller values dampen the seasonality
            # holidays_prior_scale=10.0, # Similar to seasonality_prior_scale, but for holiday effects
            # changepoint_prior_scale=0.05, # Controls how flexible the trend is; larger values allow for more flexibility, smaller values for less
            mcmc_samples=10, # If greater than 0, will use Markov Chain Monte Carlo sampling to generate forecasts, which can capture uncertainty better but is more computationally expensive
            uncertainty_samples=1000 # Number of simulated draws used to estimate uncertainty intervals
        ),

        # Deep Learning Models: Parameters dependent on data characteristics. Designed for capturing complex temporal patterns.
        'SimpleRNN': SimpleRNN(input_size, hidden_size, num_layers, output_size, dropout),
        'SimpleLSTM': SimpleLSTM(input_size, hidden_size, num_layers, output_size, dropout),
        'SimpleGRU': SimpleGRU(input_size, hidden_size, num_layers, output_size, dropout),
        'SimpleTransformer': SimpleTransformer(input_size, hidden_size, num_layers, output_size, num_heads)
    }

    return models

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Using the last time step's output
        return out

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Using the last time step's output
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Using the last time step's output
        return out

class SimpleTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads):
        super(SimpleTransformer, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        transformed = self.transformer_encoder(x)
        out = self.fc(transformed[:, -1, :])  # Using the last time step's output
        return out

# Save the model to disk
def save_model(model, filename):
    dump(model, filename)

def create_sequences(data, sequence_length=12*2):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append((seq, label))
    return sequences

def create_variable_length_sequences(data, max_sequence_length=12*2, do_shuffle=False):
    sequences = []

    for length in range(1, max_sequence_length + 1):
        for i in range(len(data) - length):
            seq = data[i:i + length]
            label = data[i + length]
            sequences.append((seq, label))

    # Optionally, shuffle the sequences to improve training
    if do_shuffle:
        random.shuffle(sequences)

    return sequences

def train_model(model, model_name, seqs, modeldir, epochs=100, learning_rate=0.0005):
    # Convert data to PyTorch tensors and move to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        # TODO: implement in batches and make sure it works over tiem
        # TODO: maybe Mlflow, TensorBoard something to track
        for seq, label in seqs:
            # Reshape sequence for model input and convert to tensor
            seq_tensor = torch.tensor(seq, dtype=torch.float32).view(-1, seq.shape[0], 1).to(device)
            label_tensor = torch.tensor(label, dtype=torch.float32).view(-1, 1).to(device)

            # Forward pass
            optimizer.zero_grad()
            prediction = model(seq_tensor)

            # Compute loss, backward pass
            loss = criterion(prediction, label_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(seqs)}')

    # Save model
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    torch.save(model.state_dict(), os.path.join(modeldir, f'{model_name}_model.pt'))

# def train_model(model, model_name, seqs, modeldir, epochs=100, learning_rate=0.0005, batch_size=32):
#     # Convert data to PyTorch tensors and move to the appropriate device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     # Prepare DataLoader for batching
#     sequences, labels = zip(*seqs)
#     sequence_tensors = torch.tensor(sequences, dtype=torch.float32).view(len(sequences), -1, 1)
#     label_tensors = torch.tensor(labels, dtype=torch.float32).view(len(labels), -1, 1)
#     dataset = TensorDataset(sequence_tensors, label_tensors)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(epochs):
#         total_loss = 0
#         for seq_batch, label_batch in dataloader:
#             # Move batch to the appropriate device
#             seq_batch = seq_batch.to(device)
#             label_batch = label_batch.to(device)

#             # Forward pass
#             optimizer.zero_grad()
#             prediction = model(seq_batch)

#             # Compute loss, backward pass
#             loss = criterion(prediction, label_batch)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

#     # Save model
#     if not os.path.exists(modeldir):
#         os.makedirs(modeldir)
#     torch.save(model.state_dict(), os.path.join(modeldir, f'{model_name}_model.pt'))