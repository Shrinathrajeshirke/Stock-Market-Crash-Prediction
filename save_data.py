import sys
import os
sys.path.append('src')

from src.components.data_ingestion import DataIngestion

def save_stock_data():
    """Save stock data to avoid API limits"""
    
    data_ingestion = DataIngestion()
    
    print("Downloading SPY data...")
    data = data_ingestion.get_stock_data('SPY', '2y')
    print(f"Saved: {data.shape[0]} rows")
    
    print("Done!")

if __name__ == "__main__":
    save_stock_data()