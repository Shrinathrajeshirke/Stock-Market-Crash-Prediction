import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf

from src.exception import CustomException
from src.utils import load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, symbol='SPY', days_ahead=7):
        """
        Make crash prediction for given symbol
        """
        try:
            model_path = os.path.join("models", "model.pkl")
            lstm_model_path = os.path.join("models", "lstm_model.h5")
            preprocessor_path = os.path.join("models", "preprocessor.pkl")
            
            # Get recent data
            data_ingestion = DataIngestion()
            df = data_ingestion.get_stock_data(symbol=symbol, period='2y')
            
            # Get last 60 days for prediction
            recent_data = df.tail(100).copy()
            
            # Define feature columns
            numerical_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'BB_middle',
                'Volatility', 'Price_Change', 'Volume_MA'
            ]
            
            # Prepare features
            features = recent_data[numerical_columns].fillna(method='ffill').fillna(method='bfill')
            
            # Load preprocessor and transform data
            if os.path.exists(preprocessor_path):
                preprocessor = load_object(file_path=preprocessor_path)
                scaled_features = preprocessor.transform(features)
            else:
                # Fallback scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
            
            # Get latest data point for prediction
            latest_features = scaled_features[-1:].reshape(1, -1)
            
            # Try LSTM model first
            crash_probability = 0.0
            model_used = "fallback"
            
            if os.path.exists(lstm_model_path):
                try:
                    # Load LSTM model
                    lstm_model = tf.keras.models.load_model(lstm_model_path)
                    
                    # Reshape for LSTM
                    n_features = len(numerical_columns)
                    if len(scaled_features) >= 60:
                        lstm_input = scaled_features[-60:].reshape(1, 60, n_features)
                        crash_probability = lstm_model.predict(lstm_input)[0][0]
                        model_used = "LSTM"
                    
                except Exception as e:
                    print(f"LSTM prediction failed: {e}")
            
            # Fallback to traditional model
            if crash_probability == 0.0 and os.path.exists(model_path):
                try:
                    model = load_object(file_path=model_path)
                    crash_probability = model.predict_proba(latest_features)[0][1]
                    model_used = "Traditional ML"
                except Exception as e:
                    print(f"Traditional model prediction failed: {e}")
            
            # Calculate additional metrics
            current_price = recent_data['Close'].iloc[-1]
            previous_price = recent_data['Close'].iloc[-2]
            price_change = ((current_price - previous_price) / previous_price) * 100
            
            # Get sentiment
            sentiment_data = data_ingestion.get_news_sentiment(symbol)
            avg_sentiment = np.mean([s['sentiment'] for s in sentiment_data[-7:]])  # Last 7 days
            
            # Risk assessment
            risk_level = "Low"
            if crash_probability > 0.7:
                risk_level = "High"
            elif crash_probability > 0.4:
                risk_level = "Medium"
            
            # Generate insights
            insights = []
            if crash_probability > 0.6:
                insights.append("High crash probability detected - consider reducing exposure")
            if avg_sentiment < -0.3:
                insights.append("Negative market sentiment observed")
            if recent_data['RSI'].iloc[-1] > 70:
                insights.append("Market appears overbought")
            elif recent_data['RSI'].iloc[-1] < 30:
                insights.append("Market appears oversold")
            
            # Volatility analysis
            volatility = recent_data['Volatility'].iloc[-1]
            avg_volatility = recent_data['Volatility'].rolling(50).mean().iloc[-1]
            volatility_status = "Normal"
            if volatility > avg_volatility * 1.5:
                volatility_status = "High"
                insights.append("Elevated volatility detected")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'crash_probability': float(crash_probability),
                'risk_level': risk_level,
                'model_used': model_used,
                'current_price': float(current_price),
                'price_change_percent': float(price_change),
                'sentiment_score': float(avg_sentiment),
                'rsi': float(recent_data['RSI'].iloc[-1]),
                'volatility': float(volatility),
                'volatility_status': volatility_status,
                'insights': insights,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 open_price: float,
                 high_price: float,
                 low_price: float,
                 close_price: float,
                 volume: float,
                 ma_5: float,
                 ma_10: float,
                 ma_20: float,
                 ma_50: float,
                 rsi: float,
                 macd: float,
                 macd_signal: float,
                 bb_upper: float,
                 bb_lower: float,
                 bb_middle: float,
                 volatility: float,
                 price_change: float,
                 volume_ma: float):
        
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.ma_5 = ma_5
        self.ma_10 = ma_10
        self.ma_20 = ma_20
        self.ma_50 = ma_50
        self.rsi = rsi
        self.macd = macd
        self.macd_signal = macd_signal
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_middle = bb_middle
        self.volatility = volatility
        self.price_change = price_change
        self.volume_ma = volume_ma
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Open": [self.open_price],
                "High": [self.high_price],
                "Low": [self.low_price],
                "Close": [self.close_price],
                "Volume": [self.volume],
                "MA_5": [self.ma_5],
                "MA_10": [self.ma_10],
                "MA_20": [self.ma_20],
                "MA_50": [self.ma_50],
                "RSI": [self.rsi],
                "MACD": [self.macd],
                "MACD_signal": [self.macd_signal],
                "BB_upper": [self.bb_upper],
                "BB_lower": [self.bb_lower],
                "BB_middle": [self.bb_middle],
                "Volatility": [self.volatility],
                "Price_Change": [self.price_change],
                "Volume_MA": [self.volume_ma]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)