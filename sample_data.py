import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_stock_data():
    """Create sample stock data for testing"""
    
    # Create 500 days of sample data
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    # Generate realistic stock price data
    np.random.seed(42)
    
    # Starting price
    price = 400.0
    prices = []
    
    for i in range(500):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% volatility
        price = price * (1 + change)
        prices.append(price)
    
    # Create OHLC data
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, 500)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, 500)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Generate volume
    volumes = np.random.randint(1000000, 10000000, 500)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    """Add technical indicators"""
    
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

def create_crash_labels(df, crash_threshold=-0.1):
    """Create crash labels"""
    # Calculate 5-day forward returns
    df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # Label crashes (1) and non-crashes (0)
    df['Crash_Label'] = (df['Future_Return'] < crash_threshold).astype(int)
    
    return df

def main():
    """Create and save sample data"""
    
    print("Creating sample stock data...")
    
    # Create sample data for different periods
    symbols = ['SPY']
    periods = ['1y', '2y', '5y']
    
    for symbol in symbols:
        for period in periods:
            print(f"Creating {symbol}_{period}.csv...")
            
            # Create data based on period
            if period == '1y':
                days = 252  # ~1 year trading days
            elif period == '2y':
                days = 504  # ~2 years
            else:
                days = 500  # Default
            
            # Create sample data
            dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
            
            # Generate data
            np.random.seed(42)
            price = 400.0
            prices = []
            
            for i in range(days):
                change = np.random.normal(0.001, 0.02)
                price = price * (1 + change)
                prices.append(price)
            
            close_prices = np.array(prices)
            high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
            low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
            open_prices = np.roll(close_prices, 1)
            open_prices[0] = close_prices[0]
            volumes = np.random.randint(1000000, 10000000, days)
            
            df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            }, index=dates)
            
            # Add indicators
            df = add_technical_indicators(df)
            df = create_crash_labels(df)
            
            # Save to data directory
            filename = f"data/{symbol}_{period}.csv"
            df.to_csv(filename)
            print(f"Saved {filename} with {len(df)} rows")
    
    print("Sample data created successfully!")
    print("You can now run: python simple_test.py")

if __name__ == "__main__":
    main()