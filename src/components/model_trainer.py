import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")
    lstm_model_path = os.path.join("models", "lstm_model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM model for time series prediction
        """
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def build_cnn_lstm_model(self, input_shape):
        """
        Build CNN-LSTM hybrid model
        """
        try:
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                Conv1D(filters=64, kernel_size=3, activation='relu'),
                Dropout(0.5),
                MaxPooling1D(pool_size=2),
                LSTM(50, return_sequences=True),
                Dropout(0.5),
                LSTM(50),
                Dropout(0.5),
                Dense(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_traditional_models(self, X_train, y_train, X_test, y_test):
        """
        Train traditional ML models
        """
        try:
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42)
            }
            
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            return model_report
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_training(self, train_array, test_array):
        """
        Initialize model training process
        """
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Train traditional models first
            logging.info("Training traditional ML models")
            model_report = self.train_traditional_models(X_train, y_train, X_test, y_test)
            
            # Get the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42)
            }
            
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            
            # Create models directory
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            
            # Save the best traditional model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Train LSTM model if data is sufficient
            if len(X_train) > 100:  # Minimum data for LSTM
                try:
                    logging.info("Training LSTM model")
                    
                    # Reshape data for LSTM (assuming sequence_length = 60)
                    n_features = 18  # Number of features
                    sequence_length = X_train.shape[1] // n_features
                    
                    if X_train.shape[1] % n_features == 0:
                        X_train_lstm = X_train.reshape(X_train.shape[0], sequence_length, n_features)
                        X_test_lstm = X_test.reshape(X_test.shape[0], sequence_length, n_features)
                        
                        # Build and train LSTM model
                        lstm_model = self.build_lstm_model((sequence_length, n_features))
                        
                        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                        )
                        
                        history = lstm_model.fit(
                            X_train_lstm, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_test_lstm, y_test),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Save LSTM model
                        lstm_model.save(self.model_trainer_config.lstm_model_path)
                        
                        # Evaluate LSTM
                        lstm_predictions = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
                        lstm_accuracy = accuracy_score(y_test, lstm_predictions)
                        
                        logging.info(f"LSTM model accuracy: {lstm_accuracy}")
                        
                        # Use LSTM model if it's better
                        if lstm_accuracy > best_model_score:
                            best_model_score = lstm_accuracy
                            logging.info("LSTM model selected as best model")
                        
                except Exception as e:
                    logging.warning(f"LSTM training failed: {str(e)}")
            
            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with decent accuracy")
            
            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)