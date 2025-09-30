# run_pipeline.py

import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    try:
        logging.info("Pipeline started")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        acc = model_trainer.initiate_model_training(train_arr, test_arr)  # <-- fixed here

        print(f"✅ Final Model Accuracy: {acc:.3f}")
        logging.info("Pipeline completed successfully")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
