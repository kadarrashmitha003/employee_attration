# src/components/model_trainer.py

import os
import sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("Splitting train/test arrays into features and target")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Training XGBoost Classifier")
            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            logging.info(f"Final Model Accuracy: {acc:.3f}")

            save_object(self.config.trained_model_file_path, model)
            logging.info(f"Model saved to {self.config.trained_model_file_path}")

            return acc

        except Exception as e:
            raise CustomException(e, sys)
