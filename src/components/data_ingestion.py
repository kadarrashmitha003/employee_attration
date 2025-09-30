# src/components/data_ingestion.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self, raw_data_path="rawdata/technova_attrition_dataset.csv"):
        self.raw_data_path = raw_data_path
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")

        try:
            df = pd.read_csv(self.raw_data_path)
            logging.info(f"Loaded raw dataset from {self.raw_data_path} with shape {df.shape}")

            os.makedirs("artifacts", exist_ok=True)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["attrition"])
            logging.info(f"Split data: train {train_df.shape}, test {test_df.shape}")

            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logging.info(f"Train/Test CSVs saved to artifacts/")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
