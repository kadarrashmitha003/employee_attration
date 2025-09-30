# src/components/data_transformation.py

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # Expected schema (exact, case-sensitive)
        self.expected_columns = [
            'age',
            'job_satisfaction',
            'salary',
            'tenure',
            'work_env_satisfaction',
            'overtime',
            'marital_status',
            'education',
            'department',
            'promotion_last_5years',
            'years_since_last_promotion',
            'training_hours',
            'work_life_balance',
            'attrition'
        ]

        # Define which columns are numeric vs categorical for preprocessing
        self.numeric_cols = [
            'age',
            'job_satisfaction',
            'salary',
            'tenure',
            'work_env_satisfaction',
            'promotion_last_5years',
            'years_since_last_promotion',
            'training_hours',
            'work_life_balance'
        ]

        self.categorical_cols = [
            'overtime',
            'marital_status',
            'education',
            'department'
        ]

        self.target_col = 'attrition'

    def _validate_and_reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and order is consistent."""
        missing = [c for c in self.expected_columns if c not in df.columns]
        if missing:
            raise CustomException(
                f"Missing required columns: {missing}. Found columns: {list(df.columns)}",
                sys
            )
        # Reorder to expected for consistency (not strictly required)
        return df[self.expected_columns]

    def _encode_target(self, s: pd.Series) -> pd.Series:
        """Robustly encode attrition to 0/1 regardless of string/boolean/numeric."""
        # If already numeric (0/1), just coerce to int
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(int)

        # Normalize strings
        s_norm = s.astype(str).str.strip().str.lower()
        mapping = {
            'yes': 1, 'y': 1, 'true': 1, '1': 1, 't': 1,
            'no': 0, 'n': 0, 'false': 0, '0': 0, 'f': 0
        }
        encoded = s_norm.map(mapping)

        # If any values couldn’t be mapped, raise a helpful error
        if encoded.isna().any():
            bad = sorted(s[encoded.isna()].unique().tolist())
            raise CustomException(
                f"attrition values not recognized (expected Yes/No, Y/N, True/False, 0/1): {bad}",
                sys
            )
        return encoded.astype(int)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, self.numeric_cols),
                ('cat', cat_pipeline, self.categorical_cols)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train and test CSVs")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Validating schema")
            train_df = self._validate_and_reorder(train_df)
            test_df = self._validate_and_reorder(test_df)

            logging.info("Encoding target column 'attrition'")
            train_df[self.target_col] = self._encode_target(train_df[self.target_col])
            test_df[self.target_col] = self._encode_target(test_df[self.target_col])

            X_train = train_df.drop(columns=[self.target_col], axis=1)
            y_train = train_df[self.target_col]
            X_test = test_df.drop(columns=[self.target_col], axis=1)
            y_test = test_df[self.target_col]

            # Ensure dtypes (some CSVs may parse numerics as object if blanks exist)
            for col in self.numeric_cols:
                if col in X_train.columns:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                if col in X_test.columns:
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting preprocessor on training features")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info(
                f"Data transformation complete. Preprocessor saved to "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
