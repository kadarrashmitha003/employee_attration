# train_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    logging.info("Pipeline start")
    ingestion = DataIngestion()
    # Path to your CSV in the repo (adjust if necessary)
    train_path, test_path = ingestion.initiate_data_ingestion(file_path='rawdata/technova_attrition.csv')

    transformer = DataTransformation()
    # target column name must match your CSV exact column (case-sensitive)
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
        target_column='Attrition'
    )

    trainer = ModelTrainer()
    score = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Training finished. Best model score: {score}")
    logging.info("Pipeline finished")
