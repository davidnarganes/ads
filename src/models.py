from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon
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

def initialize_models():

    # Set the hyperparameters
    input_size = 1  # Number of features
    hidden_size = 8  # Number of features in hidden state. Not too complex because it will overfit
    num_layers = 1  # Number of stacked layers
    output_size = 1  # Number of output time points
    num_heads = 1  # Number of heads in the Transformer model
    dropout = 0.33 # to prevent overfitting

    # Initialize statistical forecasting models
    models = {
        'NaiveForecaster': NaiveForecaster(strategy="last", sp=12),
        'ThetaForecaster': ThetaForecaster(sp=12, deseasonalize=True),
        'ExponentialSmoothing': ExponentialSmoothing(
            trend='add', damped_trend=True, seasonal='add', sp=12,
            use_boxcox=True, initialization_method='estimated',
            optimized=True, random_state=42
        ),
        'TBATS': TBATS(
            sp=[12], use_trend=True, use_box_cox=True,
            use_damped_trend=True, use_arma_errors=True,
        ),
        'ARIMA': AutoARIMA(sp=12, suppress_warnings=True),
        'SARIMAX': AutoARIMA(sp=12, seasonal=True, suppress_warnings=True),
        'Prophet': Prophet(seasonality_mode='additive'), #, yearly_seasonality=True
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
