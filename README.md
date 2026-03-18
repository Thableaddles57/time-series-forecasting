
# Time Series Forecasting

Advanced time series forecasting models (ARIMA, Prophet, LSTM) for various applications like stock prediction and demand forecasting.

## Models Implemented

- **ARIMA**: Autoregressive Integrated Moving Average model.
- **Prophet**: A forecasting procedure implemented by Facebook.
- **LSTM**: Long Short-Term Memory neural networks for sequence prediction.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Each model is implemented in its own file within the `src/` directory. You can run them directly:

```bash
python src/arima_model.py
```

## Example: ARIMA Model

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Generate some sample time series data
data = [x + random.random() * 10 for x in range(1, 100)]
dates = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
ts = pd.Series(data, index=dates)

# Fit ARIMA model
# p, d, q parameters for ARIMA (Autoregressive, Integrated, Moving Average)
model = ARIMA(ts, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast_steps = 10
forecast = model_fit.predict(start=len(ts), end=len(ts) + forecast_steps - 1)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.title('ARIMA Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
```

## Contributing

Contributions are welcome! Feel free to add more forecasting models or improve existing ones.
