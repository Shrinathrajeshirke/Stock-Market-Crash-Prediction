import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass
    
    def run_training_pipeline(self, symbol='SPY', period='5y'):
        """
        Run the complete training pipeline
        """
        try:
            logging.info("Training pipeline started")
            
            # Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
                symbol=symbol, period=period
            )
            logging.info("Data ingestion completed")
            
            # Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data transformation completed")
            
            # Model Training
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_training(train_arr, test_arr)
            logging.info("Model training completed")
            
            logging.info(f"Training pipeline completed successfully with score: {model_score}")
            
            return {
                'status': 'success',
                'model_score': model_score,
                'train_data_path': train_data_path,
                'test_data_path': test_data_path,
                'preprocessor_path': preprocessor_path
            }
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    result = pipeline.run_training_pipeline()
    print(f"Training completed with score: {result['model_score']}")