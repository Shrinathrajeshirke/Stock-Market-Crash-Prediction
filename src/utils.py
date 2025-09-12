import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]] if param else {}
            
            if para:
                gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1)
                gs.fit(X_train, y_train)
                
                model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def calculate_technical_indicators(df):
    """
    Calculate additional technical indicators
    """
    try:
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        
        # Commodity Channel Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
        
    except Exception as e:
        raise CustomException(e, sys)

def calculate_risk_metrics(returns):
    """
    Calculate various risk metrics
    """
    try:
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'volatility': returns.std() * np.sqrt(252)
        }
        
    except Exception as e:
        raise CustomException(e, sys)

def prepare_lstm_data(data, sequence_length=60):
    """
    Prepare data for LSTM model
    """
    try:
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
        
    except Exception as e:
        raise CustomException(e, sys)

def calculate_model_confidence(predictions, threshold=0.5):
    """
    Calculate model confidence based on prediction probabilities
    """
    try:
        # For binary classification
        confidence_scores = []
        
        for pred in predictions:
            if pred > threshold:
                confidence = pred
            else:
                confidence = 1 - pred
            
            confidence_scores.append(confidence)
        
        return np.array(confidence_scores)
        
    except Exception as e:
        raise CustomException(e, sys)

def format_prediction_output(prediction_data):
    """
    Format prediction output for API response
    """
    try:
        formatted_output = {
            'prediction': float(prediction_data.get('crash_probability', 0)),
            'risk_level': prediction_data.get('risk_level', 'Unknown'),
            'confidence': float(prediction_data.get('confidence', 0)),
            'market_indicators': {
                'rsi': float(prediction_data.get('rsi', 0)),
                'volatility': float(prediction_data.get('volatility', 0)),
                'sentiment': float(prediction_data.get('sentiment_score', 0))
            },
            'recommendations': prediction_data.get('insights', []),
            'timestamp': prediction_data.get('prediction_date', ''),
            'model_used': prediction_data.get('model_used', 'Unknown')
        }
        
        return formatted_output
        
    except Exception as e:
        raise CustomException(e, sys)