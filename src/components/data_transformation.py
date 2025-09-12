import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('models', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        Create the preprocessing pipeline
        """
        try:
            # Define feature columns
            numerical_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'BB_middle',
                'Volatility', 'Price_Change', 'Volume_MA'
            ]
            
            # Create preprocessing pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_sequences(self, data, target, sequence_length=60):
        """
        Create sequences for LSTM model
        """
        try:
            X, y = [], []
            
            for i in range(sequence_length, len(data)):
                X.append(data[i-sequence_length:i])
                y.append(target[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initialize data transformation
        """
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Crash_Label"
            
            # Define feature columns
            numerical_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'BB_middle',
                'Volatility', 'Price_Change', 'Volume_MA'
            ]
            
            # Separate features and target
            input_feature_train_df = train_df[numerical_columns]
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df[numerical_columns]
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing datasets")
            
            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Create sequences for LSTM
            sequence_length = 60
            
            if len(input_feature_train_arr) > sequence_length:
                X_train, y_train = self.create_sequences(
                    input_feature_train_arr, 
                    target_feature_train_df.values, 
                    sequence_length
                )
            else:
                # For smaller datasets, use all data without sequences
                X_train = input_feature_train_arr
                y_train = target_feature_train_df.values
            
            if len(input_feature_test_arr) > sequence_length:
                X_test, y_test = self.create_sequences(
                    input_feature_test_arr, 
                    target_feature_test_df.values, 
                    sequence_length
                )
            else:
                X_test = input_feature_test_arr
                y_test = target_feature_test_df.values
            
            # Combine arrays
            train_arr = np.c_[X_train.reshape(X_train.shape[0], -1), y_train]
            test_arr = np.c_[X_test.reshape(X_test.shape[0], -1), y_test]
            
            logging.info("Saved preprocessing object")
            
            # Create models directory
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)