from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Import custom modules
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_object

app = Flask(__name__)

# Global variables for caching
model_cache = {}
data_cache = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Initialize components
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer()
        
        # Get training parameters
        symbol = request.json.get('symbol', 'SPY')
        period = request.json.get('period', '5y')
        
        # Data ingestion
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
            symbol=symbol, period=period
        )
        
        # Data transformation
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        # Model training
        model_score = model_trainer.initiate_model_training(train_arr, test_arr)
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained successfully for {symbol}',
            'accuracy': f"{model_score:.4f}",
            'symbol': symbol
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.json.get('symbol', 'SPY')
        days_ahead = request.json.get('days_ahead', 7)
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Make prediction
        result = predict_pipeline.predict(symbol=symbol, days_ahead=days_ahead)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        })

@app.route('/get_market_data', methods=['GET'])
def get_market_data():
    try:
        symbol = request.args.get('symbol', 'SPY')
        
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Get recent market data
        market_data = data_ingestion.get_recent_data(symbol=symbol, period='1y')
        
        # Convert to JSON serializable format
        market_data_json = market_data.tail(100).to_dict(orient='records')
        
        # Add dates as string
        for i, record in enumerate(market_data_json):
            record['Date'] = market_data.index[-100:][i].strftime('%Y-%m-%d')
        
        return jsonify({
            'status': 'success',
            'data': market_data_json,
            'symbol': symbol
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get market data: {str(e)}'
        })

@app.route('/get_sentiment', methods=['GET'])
def get_sentiment():
    try:
        symbol = request.args.get('symbol', 'SPY')
        
        # Initialize data ingestion for sentiment
        data_ingestion = DataIngestion()
        
        # Get sentiment analysis
        sentiment_data = data_ingestion.get_news_sentiment(symbol=symbol)
        
        return jsonify({
            'status': 'success',
            'sentiment': sentiment_data,
            'symbol': symbol
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get sentiment: {str(e)}'
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Production settings for Render
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False") == "True"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)