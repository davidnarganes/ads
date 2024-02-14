import pandas as pd
import torch
import numpy as np
from models import (
    initialize_models, 
    train_model,
    save_model,
    create_sequences,
    initialize_random_seed
    )

# Function to perform predictions using sktime compatible models
def forecast_and_evaluate(forecaster, y_train, num_predictions):
    forecaster.fit(y_train)
    # Generate predictions for the next `num_predictions` periods
    fh = np.arange(1, num_predictions + 1)
    y_pred = forecaster.predict(fh)
    return y_pred

# Function to perform predictions using PyTorch models
def forecast_and_evaluate_pytorch(model, y_train, num_predictions, max_sequence_length=24):
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepare the initial sequence from the end of y_train
    initial_sequence = y_train[-max_sequence_length:].values if len(y_train) > max_sequence_length else y_train.values
    
    # Predictions
    y_pred = []
    current_sequence = np.array([initial_sequence])
    
    with torch.no_grad():
        for _ in range(num_predictions):
            seq_tensor = torch.tensor(current_sequence, dtype=torch.float32).to(device)
            output = model(seq_tensor.unsqueeze(-1))
            next_value = output.cpu().numpy().flatten()[0]  # Get the next predicted value
            y_pred.append(next_value)
            # Update the sequence with the predicted value
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1] = next_value
    
    return np.array(y_pred)

# Modify the main code to produce predictions without a test set
outdir = '../pics'
modeldir = '../models'
df = pd.read_csv('../data/train.csv')
df.columns = ['date', 'y']
df.dropna(how='all', inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%y')
df.sort_values(by='date', inplace=True)
y_train = df['y']

# Initialize models
initialize_random_seed(123)
forecasters = initialize_models()

# Train and generate predictions for each model
results = []
for model_name, forecaster in forecasters.items():
    if 'sktime' in str(type(forecaster)):
        y_pred = forecast_and_evaluate(forecaster, y_train, 12)
    else:
        seqs = create_sequences(y_train.values, sequence_length=24)
        train_model(forecaster, model_name, seqs, modeldir, epochs=100, learning_rate=0.001)
        y_pred = forecast_and_evaluate_pytorch(forecaster, y_train, 12, max_sequence_length=24)
    
    # Convert predictions to DataFrame
    y_pred = pd.DataFrame(y_pred, columns=['y'])
    y_pred['date'] = pd.date_range(start='2021-03-01', periods=12, freq='MS')
    
    # Append results
    results.append({
        'model_name': model_name,
        'predictions': y_pred['y'].values.tolist(),
        'dates': y_pred['date'].dt.strftime('%d.%m.%Y').values.tolist()
    })

    print(model_name, forecaster)
    
    # Save the forecaster
    save_model(forecaster, f'{modeldir}/{model_name}_forecaster.joblib')

# Output predictions to a CSV file
results_df = pd.DataFrame()
for result in results:
    predictions_series = pd.Series(result['predictions'])
    dates_series = pd.to_datetime(result['dates'], format='%d.%m.%Y')
    model_df = pd.DataFrame({
        'model_name': result['model_name'],
        'date': dates_series,
        'predictions': predictions_series
    })
    results_df = pd.concat([results_df, model_df], ignore_index=True)

results_wide_df = results_df.pivot_table(index='date', columns='model_name', values='predictions').reset_index()
results_wide_df.to_csv('../output/test.csv', index=False)
print(results_wide_df)
