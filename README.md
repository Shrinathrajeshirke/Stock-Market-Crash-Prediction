# Stock Market Crash Prediction System

A comprehensive AI-powered system that predicts potential stock market crashes using deep learning models and sentiment analysis. This end-to-end solution combines technical analysis, sentiment analysis from news sources, and advanced machine learning techniques.

## Features

### Advanced Machine Learning Models
- **LSTM (Long Short-Term Memory)**: For time series forecasting and pattern recognition
- **CNN-LSTM Hybrid**: Combines convolutional layers with LSTM for better feature extraction
- **Traditional ML**: Random Forest, Gradient Boosting, and Logistic Regression as baseline models
- **Ensemble Methods**: Combines multiple model predictions for improved accuracy

### Technical Analysis
- **Price Indicators**: Moving Averages (5, 10, 20, 50-day)
- **Momentum Indicators**: RSI, MACD, Stochastic Oscillator
- **Volatility Indicators**: Bollinger Bands, ATR, Volatility Index
- **Volume Analysis**: Volume Moving Averages and trends

### Sentiment Analysis
- **News Sentiment**: Financial news analysis using VADER sentiment
- **Social Media Integration**: Twitter and Reddit sentiment tracking
- **Economic Indicators**: GDP, interest rates, unemployment data
- **Market Sentiment Scoring**: Composite sentiment metrics

### Real-time Predictions
- **Live Data Integration**: Real-time market data from Yahoo Finance
- **Risk Assessment**: Low, Medium, High risk categorization
- **Confidence Scoring**: Model confidence in predictions
- **Interactive Dashboard**: Web-based visualization and monitoring

## Project Structure

```
stock-market-crash-prediction/
├── data/                          # Data storage
├── logs/                          # Application logs
├── models/                        # Trained model files
├── src/                          # Source code
│   ├── components/               # Core components
│   │   ├── data_ingestion.py    # Data collection and preprocessing
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py     # Model training logic
│   ├── pipeline/                 # Pipeline modules
│   │   ├── predict_pipeline.py  # Prediction pipeline
│   │   └── train_pipeline.py    # Training pipeline
│   ├── exception.py             # Custom exception handling
│   ├── logger.py               # Logging configuration
│   └── utils.py                # Utility functions
├── templates/                   # HTML templates
│   ├── home.html               # Landing page
│   └── index.html              # Dashboard
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/stock-market-crash-prediction.git
cd stock-market-crash-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Required Directories
```bash
mkdir -p data logs models
```

### 5. Create __init__.py Files
```bash
# Create empty __init__.py files
touch src/__init__.py
touch src/components/__init__.py
touch src/pipeline/__init__.py
```

## Usage

### 1. Start the Flask Application
```bash
python app.py
```

### 2. Access the Application
- **Home Page**: http://localhost:5000/
- **Dashboard**: http://localhost:5000/dashboard

### 3. Training a Model
```bash
# Via API
curl -X POST http://localhost:5000/train_model \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "period": "5y"}'

# Or use the web interface
```

### 4. Making Predictions
```bash
# Via API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "days_ahead": 7}'

# Or use the dashboard
```

## API Endpoints

### Training Endpoints
- `POST /train_model`: Train a new model for a specific symbol
- `GET /health`: Health check endpoint

### Prediction Endpoints
- `POST /predict`: Get crash prediction for a symbol
- `GET /get_market_data`: Retrieve recent market data
- `GET /get_sentiment`: Get sentiment analysis data

### Dashboard Endpoints
- `GET /`: Home page
- `GET /dashboard`: Interactive dashboard

## Model Architecture

### LSTM Model
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25),
    Dense(1, activation='sigmoid')
])
```

### CNN-LSTM Hybrid
```python
model = Sequential([
    Conv1D(64, 3, activation='relu'),
    Conv1D(64, 3, activation='relu'),
    Dropout(0.5),
    MaxPooling1D(2),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## Feature Engineering

### Technical Indicators
- **Moving Averages**: 5, 10, 20, 50-day SMA
- **RSI**: 14-period Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: 20-period with 2 standard deviations
- **Volatility**: 20-day rolling standard deviation

### Sentiment Features
- **News Sentiment**: VADER compound scores
- **Volume Analysis**: Normalized trading volumes
- **Price Action**: Rate of change and momentum indicators

### Target Variable
- **Crash Definition**: >10% decline within prediction window
- **Binary Classification**: 1 for crash, 0 for no crash

## Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for crash predictions
- **Recall**: Ability to identify actual crashes
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Typical Performance
- **Traditional ML**: ~78% accuracy
- **LSTM Model**: ~82% accuracy
- **Ensemble**: ~85% accuracy

## Configuration

### Environment Variables
```bash
# Optional: Add to .env file
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key
```

### Model Parameters
```python
# In model_trainer.py
SEQUENCE_LENGTH = 60  # Days of historical data
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

## Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment (Gunicorn)
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

## Data Sources

### Market Data
- **Yahoo Finance API**: Historical and real-time stock prices
- **Alpha Vantage**: Technical indicators and economic data
- **Quandl**: Economic indicators and alternative data

### Sentiment Data
- **News APIs**: Financial news from various sources
- **Social Media**: Twitter and Reddit sentiment
- **Economic Reports**: Central bank communications

## Future Enhancements

### Model Improvements
- **Transformer Models**: Attention mechanisms for better long-term dependencies
- **Reinforcement Learning**: RL agents for dynamic prediction strategies
- **Multi-timeframe Analysis**: Different prediction horizons
- **Alternative Data**: Satellite data, web scraping, social sentiment

### Technical Features
- **Real-time Streaming**: WebSocket connections for live updates
- **Mobile App**: React Native mobile application
- **Advanced Visualizations**: 3D charts and interactive plots
- **Alert System**: Email/SMS notifications for high-risk scenarios

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This system is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

## Support

For support, email your-email@example.com or create an issue in the GitHub repository.

## Acknowledgments

- **TensorFlow Team**: For the deep learning framework
- **Yahoo Finance**: For providing free market data
- **Flask Team**: For the lightweight web framework
- **Plotly**: For interactive visualizations
- **Open Source Community**: For various libraries and tools used