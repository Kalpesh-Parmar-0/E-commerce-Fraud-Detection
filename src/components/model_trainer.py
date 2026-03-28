import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object, evaluate_model

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
            logging.info("splitting to X_train, y_train, X_val, y_val")
            X_train, y_train, X_val, y_val = (
                train_array[:, :-1],
                train_array[:, -1],
                val_array[:, :-1],
                val_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                "XGB Classifier": XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=1),
                "LGB Classifier": LGBMClassifier(class_weight = 'balanced', random_state=42, n_jobs=1)
            }

            params = {
                "Random Forest":{
                    'n_estimators':[100],
                    'max_features': ['sqrt'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "XGB Classifier": {
                    'n_estimators': [100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [6, 10],
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

            logging.info("evaluating hyprparameter tuning")
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, models=models, params=params)
            
            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model_score = model_report[best_model_name]["score"]
            best_model = models[best_model_name]["model"]

            logging.info(f"Best model found, model name is {best_model_name}, accuracy score is {best_model_score}")
            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)
        except Exception as e:
            raise CustomeException(e, sys)