import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def visualize_time_series(df):
    plt.figure(figsize=(20, 6))  

    # Scatter plot of data points
    years = df['date'].dt.year
    plt.scatter(df['date'], df['y'], c=years, cmap='Set1', label='Values', s=70)
    plt.plot(df['date'], df['y'], label='Line', color='black')

    # Calculate and plot moving averages
    quarterly_window = 3
    yearly_window = 12
    plt.plot(df['date'], df['y'].rolling(window=quarterly_window).mean(), 
             color='gold', label=f'Quarterly Rolling Mean (w={quarterly_window})')
    plt.plot(df['date'], df['y'].rolling(window=yearly_window).mean(), 
             color='crimson', ls='--', label=f'Yearly Rolling Mean (w={yearly_window})')

    # Highlighting seasons
    for year in np.unique(years):
        for quarter in range(1, 5):
            start_date = pd.Timestamp(f'{year}-Q{quarter}')
            end_date = start_date + pd.offsets.QuarterEnd()
            plt.axvspan(start_date, end_date, color='grey', alpha=0.1)

    # Set titles and labels
    plt.title('Time Series Data with Trends and Seasonality', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)

    # Format the dates on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()

    plt.grid(True)
    plt.tick_params(labelsize=9)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

def plot_predictions(df, forecasters, last_idx, show_fig=False):
    num_forecasters = len(forecasters)
    fig, axes = plt.subplots(nrows=num_forecasters, ncols=1, figsize=(12, 2 * num_forecasters), sharex=True)

    # Ensure axes is always a list for consistency
    if num_forecasters == 1:
        axes = [axes]

    for model_name, ax in zip(forecasters, axes):

        ax.set_ylabel('Y vals')

        # Plot actual data
        ax.plot(df['date'], df['y'], label='Actual', color='blue', lw=2)
        ax.scatter(df['date'][:last_idx], df['y'][:last_idx], s=10, color='blue')

        # Calculate and plot rolling average
        rolling_avg = df['y'].rolling(window=3, min_periods=1).mean()
        ax.plot(df['date'][:last_idx], rolling_avg[:last_idx], label='3-pt Rolling Avg', color='green', linestyle='--')

        # Plot predicted data
        if last_idx < -2:
            ax.plot(df['date'][last_idx:-2], df[f'pred_{model_name}'][last_idx:-2], label=f'Predicted {model_name}', color='black')
            ax.scatter(df['date'][last_idx:-2], df[f'pred_{model_name}'][last_idx:-2], s=10, color='black')
        else:
            ax.plot(df['date'][last_idx:], df[f'pred_{model_name}'][last_idx:], label=f'Predicted {model_name}', color='black')
            ax.scatter(df['date'][last_idx:], df[f'pred_{model_name}'][last_idx:], s=10, color='black')

        # Plot future data
        if last_idx < -2:
            ax.plot(df['date'][last_idx:-2], df['y'][last_idx:-2], label='Actual (Future)', color='crimson')
            ax.scatter(df['date'][last_idx:-2], df['y'][last_idx:-2], s=20, color='crimson')
        else:
            ax.plot(df['date'][last_idx:], df['y'][last_idx:], label='Actual (Future)', color='crimson')
            ax.scatter(df['date'][last_idx:], df['y'][last_idx:], s=20, color='crimson')        

        # Add legend to the plot
        ax.legend(loc='upper left')

    plt.tight_layout()
    if show_fig:
         plt.show()
    return fig
   