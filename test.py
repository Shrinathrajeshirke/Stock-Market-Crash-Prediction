import sys
import os

# Add src to path
sys.path.append('src')

# Create directories if needed
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Create __init__.py files if needed
open('src/__init__.py', 'a').close()
open('src/components/__init__.py', 'a').close()
open('src/pipeline/__init__.py', 'a').close()

print("Testing imports...")

try:
    from src.components.data_ingestion import DataIngestion
    print("DataIngestion imported")
except Exception as e:
    print(f"DataIngestion import failed: {e}")
    exit(1)

try:
    from src.components.data_transformation import DataTransformation
    print("DataTransformation imported")
except Exception as e:
    print(f"DataTransformation import failed: {e}")
    exit(1)

try:
    from src.components.model_trainer import ModelTrainer
    print("ModelTrainer imported")
except Exception as e:
    print(f"ModelTrainer import failed: {e}")
    exit(1)

try:
    from src.pipeline.predict_pipeline import PredictPipeline
    print("PredictPipeline imported")
except Exception as e:
    print(f"PredictPipeline import failed: {e}")
    exit(1)

print("All imports successful")

print("\nTesting data ingestion...")
data_ingestion = DataIngestion()
df = data_ingestion.get_stock_data('SPY', '1y')
print(f"Got data shape: {df.shape}")

print("\nTesting full data ingestion pipeline...")
train_path, test_path = data_ingestion.initiate_data_ingestion('SPY', '2y')
print(f"Train file: {train_path}")
print(f"Test file: {test_path}")

print("\nTesting data transformation...")
data_transformation = DataTransformation()
train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
print(f"Train array shape: {train_arr.shape}")
print(f"Test array shape: {test_arr.shape}")

print("\nTesting model training...")
model_trainer = ModelTrainer()
score = model_trainer.initiate_model_training(train_arr, test_arr)
print(f"Model score: {score}")

print("\nTesting prediction...")
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict('SPY', 7)
print(f"Prediction result: {result['crash_probability']}")
print(f"Risk level: {result['risk_level']}")

print("\nAll tests completed successfully!")