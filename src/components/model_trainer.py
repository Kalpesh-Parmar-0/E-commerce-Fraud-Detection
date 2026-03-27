import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "XGB Classifier": XGBClassifier(),
                "LGB Classifier": LGBMClassifier(class_weight = 'balanced')
            }

            params = {
                "Random Forest":{
                    'n_estimators':[100, 200],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "XGB Classifier": {
                    'class_weight': ['balanced'],
                    'n_estimators': [100, 300],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [4, 6, 10],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'scale_pos_weight': [1, 5, 10]
                },
                "LGB Classifier": {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [31, 64],
                    'max_depth': [-1, 10],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'scale_pos_weight': [1, 5, 10]   # IMPORTANT for imbalance
                }
            }
        except Exception as e:
            raise CustomeException(e, sys)