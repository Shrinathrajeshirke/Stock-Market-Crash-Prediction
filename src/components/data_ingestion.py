import os
import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'train.csv')
    test_data_path: str = os.path.join('data', 'test.csv')
    raw_data_path: str = os.path.join('data', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.analyzer = SentimentIntensityAnalyzer()
    
    def get_stock_data(self, symbol='SPY', period='5y'):
        """
        Fetch stock data using yfinance with simple caching
        """
        try:
            # Simple cache file name
            cache_file = "data/" + symbol + "_" + period + ".csv"
            
            # If file exists, use it
            if os.path.exists(cache_file):
                logging.info("Using cached data for " + symbol)
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return data
            
            # If no cache, fetch from API
            logging.info("Fetching stock data for " + symbol)
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                raise CustomException("No data found for symbol " + symbol)
            
            # Add technical indicators
            data = self.add_technical_indicators(data)
            
            # Save for next time
            data.to_csv(cache_file)
            logging.info("Data saved to " + cache_file)
            
            return data
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        """
        try:
            # Moving Averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # Volatility
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Price change
            df['Price_Change'] = df['Close'].pct_change()
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_news_sentiment(self, symbol='SPY'):
        """
        Get news sentiment for the given symbol
        """
        try:
            logging.info("Getting news sentiment for " + symbol)
            
            # Simple news sentiment (using placeholder data for demo)
            # In production, you would integrate with news APIs
            sentiment_scores = []
            
            # Generate sample sentiment data
            np.random.seed(42)
            for i in range(30):  # Last 30 days
                date = datetime.now() - timedelta(days=i)
                sentiment_score = np.random.uniform(-1, 1)  # Random sentiment between -1 and 1
                sentiment_scores.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'sentiment': sentiment_score,
                    'compound': sentiment_score
                })
            
            logging.info("News sentiment data generated")
            return sentiment_scores
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_crash_labels(self, df, crash_threshold=-0.1):
        """
        Create crash labels based on significant price drops
        """
        try:
            # Calculate 5-day forward returns
            df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
            
            # Label crashes (1) and non-crashes (0)
            df['Crash_Label'] = (df['Future_Return'] < crash_threshold).astype(int)
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_recent_data(self, symbol='SPY', period='1y'):
        """
        Get recent data for visualization
        """
        try:
            return self.get_stock_data(symbol, period)
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self, symbol='SPY', period='5y'):
        """
        Initialize data ingestion process
        """
        logging.info("Entered the data ingestion method")
        
        try:
            # Create data directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Get stock data
            df = self.get_stock_data(symbol, period)
            
            # Create crash labels
            df = self.create_crash_labels(df)
            
            # Remove NaN values
            df = df.dropna()
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path)
            logging.info("Raw data saved")
            
            # Split data (80-20 split)
            train_size = int(len(df) * 0.8)
            train_df = df[:train_size]
            test_df = df[train_size:]
            
            # Save train and test data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()