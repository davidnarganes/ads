import pandas as pd
import numpy as np
from metrics import calculate_metrics
from plotting import plot_predictions
from models import (
    initialize_models, 
    forecast_and_evaluate, 
    forecast_and_evaluate_pytorch,
    save_model,
    create_sequences,
    create_variable_length_sequences,
    train_model
    )

outdir = '../pics'
modeldir = '../models'
df = pd.read_csv('../data/train.csv')
df.columns = ['date','y']
df.dropna(how='all', inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%y')
df.sort_values(by='date', inplace=True)

# Regress the values of the last year, ignoring Jan and Feb
use_feats = ['y','month','year']
use_feats = ['y']
last_idxs = [-14-12, -14, -2]

for last_idx in last_idxs:

    y_train = df[:last_idx][use_feats]
    if last_idx < -2:
        y_test = df[last_idx:-2][use_feats]
    else:
        y_test = df[last_idx:][use_feats]

    # Models
    forecasters = initialize_models()

    results = []
    for model_name, forecaster in forecasters.items():

        if 'sktime' in str(type(forecaster)):
            n_periods = len(y_test)
            y_pred, forecaster = forecast_and_evaluate(forecaster, y_train, y_test)
        else:
            seqs = create_sequences(y_train.values)
            # seqs = create_variable_length_sequences(y_train.values)
            print(f'There are {len(seqs)} sequences to train on')
            train_model(forecaster, model_name, seqs, modeldir)
            y_pred = forecast_and_evaluate_pytorch(forecaster, y_train, y_test)
            y_pred = pd.DataFrame(y_pred, columns=['y'], index=y_test.index)

        metrics_1st_year = calculate_metrics(y_test[:12], y_pred[:12])
        metrics_2nd_year = calculate_metrics(y_test[12:], y_pred[12:])
        metrics = calculate_metrics(y_test, y_pred)

        y_train = df[:last_idx][use_feats]
        if last_idx < -2:
            dates = df[last_idx:-2].index
        else:
            dates = df[last_idx:].index
        
        df[f'pred_{model_name}'] = np.nan
        df.loc[dates, f'pred_{model_name}'] = y_pred['y'].tolist()
        
        print(f"{model_name}: {metrics}")
        
        # Rename keys. I suspect there is something going on with the prediction for the prelast and last 12 months
        metrics_1st_year_renamed = {f'first_12_{key}': value for key, value in metrics_1st_year.items()}
        metrics_2nd_year_renamed = {f'second_12_{key}': value for key, value in metrics_2nd_year.items()}
        
        # Append metrics to results list
        results.append({
            'model_name': model_name,
            **metrics_1st_year_renamed,  
            **metrics_2nd_year_renamed,
            **metrics
        })

        # Save the forecaster
        save_model(forecaster, f'{modeldir}/{model_name}_forecaster_{last_idx}.joblib')

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{outdir}/results_last_{last_idx}.csv', index=False)

    # Call the function with the dataframe and forecasters
    fig = plot_predictions(df, forecasters, last_idx, show_fig=False)
    fig.savefig(f'{outdir}/predictions_last_{last_idx}.pdf')