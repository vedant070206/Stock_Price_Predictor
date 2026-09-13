# Microsoft Stock Price Predictor

A deep learning project that predicts Microsoft (MSFT) stock prices using a Long Short-Term Memory (LSTM) neural network trained on historical daily OHLCV data.

This project demonstrates how time-series forecasting can be applied to financial data using Python, TensorFlow, and scikit-learn. It includes data preparation, feature scaling, sequential modeling, prediction generation, and visualization of actual versus predicted stock prices.

## Overview

The goal of this project is to forecast the future trend of Microsoft's closing price using historical market data. The model uses a sliding-window approach to learn temporal dependencies from previous price movements and predicts the next closing value based on a 60-day lookback window.

Key objectives:
- Load and clean historical stock market data
- Explore trends, volume patterns, and correlations
- Scale data for neural network compatibility
- Train an LSTM model with time-series sequences
- Evaluate prediction quality against actual values
- Visualize results clearly for analysis

## Project Structure

```text
Stock Price Predictor/
├── model.ipynb          # Main notebook with preprocessing, model training, and visualization
├── MSFT.csv             # Historical Microsoft stock data
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Optional local ignore rules if added later
```

## Dataset Details

The project uses a daily Microsoft stock dataset stored in `MSFT.csv`.

### Features
- `Date`: Trading date
- `Open`: Opening price
- `High`: Highest price of the day
- `Low`: Lowest price of the day
- `Close`: Closing price
- `Adj Close`: Adjusted closing price
- `Volume`: Number of shares traded

### Data Characteristics
- Time range: 1986 to recent years (dataset includes long-horizon historical stock performance)
- Granularity: Daily market data
- Use case: Time-series regression / stock forecasting

### Data Preparation
The notebook performs the following steps:
- Converts the `Date` column to datetime format
- Selects the closing price as the target feature
- Uses a 60-day window for sequence generation
- Scales the dataset using `StandardScaler`
- Splits the dataset into training and testing sequences

## Methodology

### Model Type
This project uses an LSTM neural network, which is well-suited for sequential data and temporal patterns in stock prices.

### Architecture
The model is built as a `Sequential` Keras network with:
- LSTM layer with 64 units and `return_sequences=True`
- Second LSTM layer with 64 units and `return_sequences=False`
- Dense layer with 128 units and ReLU activation
- Dropout layer for regularization
- Final Dense output layer predicting one value

### Training Setup
- Target: Close price
- Input sequence length: 60 days
- Training data proportion: approximately 95%
- Optimizer: Adam
- Loss function: Mean Absolute Error (`mae`)
- Additional metric: Root Mean Squared Error (`RootMeanSquaredError`)
- Epochs: 20
- Batch size: 32

## Metrics and Evaluation

The model is configured to track metrics relevant to stock price prediction:

- `MAE` (Mean Absolute Error): measures average absolute difference between actual and predicted closing prices
- `RMSE` (Root Mean Squared Error): emphasizes larger prediction errors

The notebook also includes visual evaluation by plotting:
- Train vs actual close prices
- Test actual close prices
- Predicted close prices
- Trend comparison across time

This makes it easier to interpret how well the model captures market direction and price movement patterns.

## Dependencies

The project uses the following libraries:

```bash
tensorflow==2.21.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd "Stock Price Predictor"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open `model.ipynb` in Jupyter Notebook or VS Code and run all cells sequentially.

## Usage

1. Ensure `MSFT.csv` is in the project root.
2. Open the notebook.
3. Execute the cells in order.
4. Review the exploratory data analysis and model training output.
5. Inspect the final prediction plot comparing actual and predicted values.

## Key Takeaways

This project is a practical example of applying deep learning to financial forecasting and demonstrates:
- time-series preprocessing for stock data
- LSTM-based sequence learning
- model evaluation using regression metrics
- effective result visualization for business insights

## Future Improvements

Possible enhancements for a more production-ready version include:
- Adding more technical indicators such as SMA, EMA, RSI, and MACD
- Expanding the feature set beyond closing price alone
- Using a more robust validation strategy such as walk-forward validation
- Experimenting with bidirectional LSTMs, GRUs, or Transformer-based models
- Comparing against baseline models like ARIMA or XGBoost
- Deploying the model as a Streamlit dashboard for live predictions

## Notes

This project is intended for learning, experimentation, and educational stock prediction analysis. Financial forecasting is inherently uncertain, so the model should be treated as a research and demonstration tool rather than financial advice.

## Contact

- Email: thakar2006@gmail.com
- LinkedIn: https://www.linkedin.com/in/vedant-thakar-4ba561292/
- GitHub: https://github.com/vedant070206


