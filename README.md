# Readme

## Overview
This project is structured to facilitate the development, training, and evaluation of time series forecasting models. It encompasses data preprocessing, model implementation, training routines, result visualization, and documentation.

## Directory Structure

- `data`: Contains the training dataset `train.csv` which is used to train and evaluate the forecasting models.
- `docs`: Includes the project documentation, such as the `case problem.pdf` which outlines the problem statement and objectives.
- `environment.yml`: A YAML file specifying the Python environment required to run the project, ensuring consistency across different setups.
- `models`: Stores serialized versions of trained models (`*.joblib` for scikit-learn compatible models and `*.pt` for PyTorch models) at various checkpoints indicated by the suffixes (`-14`, `-2`, `-26`).
- `pics`: Holds visual outputs of the models' predictions and performance metrics as PDFs and CSVs, allowing for easy reference and comparison.
- `src`: Contains all source code files including the main Jupyter notebook (`main.ipynb`), Python modules for metrics (`metrics.py`), model definitions (`models.py`), plotting utilities (`plotting.py`), and additional notebooks for specific analyses (e.g., `which_seasonality_concerns_for_2021.ipynb`).

The structure:
```
.
├── README.md
├── data
│   └── train.csv
├── docs
│   └── case problem.pdf
├── environment.yml
├── models
│   ├── ARIMA_forecaster.joblib
│   ├── ARIMA_forecaster_-14.joblib
│   ├── ARIMA_forecaster_-2.joblib
│   ├── ARIMA_forecaster_-26.joblib
│   ├── ExponentialSmoothing_forecaster.joblib
│   ├── ExponentialSmoothing_forecaster_-14.joblib
│   ├── ExponentialSmoothing_forecaster_-2.joblib
│   ├── ExponentialSmoothing_forecaster_-26.joblib
│   ├── NaiveForecaster_forecaster.joblib
│   ├── NaiveForecaster_forecaster_-14.joblib
│   ├── NaiveForecaster_forecaster_-2.joblib
│   ├── NaiveForecaster_forecaster_-26.joblib
│   ├── Prophet_forecaster.joblib
│   ├── Prophet_forecaster_-14.joblib
│   ├── Prophet_forecaster_-2.joblib
│   ├── Prophet_forecaster_-26.joblib
│   ├── RNN_model.pt
│   ├── SARIMAX_forecaster.joblib
│   ├── SARIMAX_forecaster_-14.joblib
│   ├── SARIMAX_forecaster_-2.joblib
│   ├── SARIMAX_forecaster_-26.joblib
│   ├── SimpleGRU_forecaster.joblib
│   ├── SimpleGRU_forecaster_-14.joblib
│   ├── SimpleGRU_forecaster_-2.joblib
│   ├── SimpleGRU_forecaster_-26.joblib
│   ├── SimpleGRU_model.pt
│   ├── SimpleLSTM_forecaster.joblib
│   ├── SimpleLSTM_forecaster_-14.joblib
│   ├── SimpleLSTM_forecaster_-2.joblib
│   ├── SimpleLSTM_forecaster_-26.joblib
│   ├── SimpleLSTM_model.pt
│   ├── SimpleRNN_forecaster.joblib
│   ├── SimpleRNN_forecaster_-14.joblib
│   ├── SimpleRNN_forecaster_-2.joblib
│   ├── SimpleRNN_forecaster_-26.joblib
│   ├── SimpleRNN_model.pt
│   ├── SimpleTransformer_forecaster.joblib
│   ├── SimpleTransformer_forecaster_-14.joblib
│   ├── SimpleTransformer_forecaster_-2.joblib
│   ├── SimpleTransformer_forecaster_-26.joblib
│   ├── SimpleTransformer_model.pt
│   ├── TBATS_forecaster.joblib
│   ├── TBATS_forecaster_-14.joblib
│   ├── TBATS_forecaster_-2.joblib
│   ├── TBATS_forecaster_-26.joblib
│   ├── ThetaForecaster_forecaster.joblib
│   ├── ThetaForecaster_forecaster_-14.joblib
│   ├── ThetaForecaster_forecaster_-2.joblib
│   └── ThetaForecaster_forecaster_-26.joblib
├── output
│   └── test.csv
├── pics
│   ├── predictions_last_-14.pdf
│   ├── predictions_last_-2.pdf
│   ├── predictions_last_-26.pdf
│   ├── results_last_-14.csv
│   ├── results_last_-2.csv
│   └── results_last_-26.csv
└── src
    ├── __pycache__
    │   ├── metrics.cpython-310.pyc
    │   ├── metrics.cpython-39.pyc
    │   ├── models.cpython-310.pyc
    │   ├── models.cpython-39.pyc
    │   ├── plotting.cpython-310.pyc
    │   └── plotting.cpython-39.pyc
    ├── main.ipynb
    ├── metrics.py
    ├── models.py
    ├── plotting.py
    ├── train_and_validate.py
    ├── train_predict_for_2022.py
    └── which_seasonality_concerns_for_2021.ipynb

8 directories, 73 files
```

## Usage

To utilize the project:
1. Set up the Python environment using the `environment.yml` file to ensure all dependencies are correctly installed.
2. Explore the `main.ipynb` notebook for an overview of the project workflow, including data loading, preprocessing, model training, and evaluation.
3. For a detailed analysis of seasonality impacts on model performance, particularly for the year 2021, refer to the `which_seasonality_concerns_for_2021.ipynb` notebook.
4. The `models` directory can be used to retrieve trained models for further analysis or deployment.
5. Visual results are stored in the `pics` directory, offering a graphical representation of model performance over time.
6. The Python modules in the `src` directory provide all the necessary functionalities to support the notebooks and can be modified for custom requirements.