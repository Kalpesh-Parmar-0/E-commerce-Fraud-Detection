import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object, evaluate_model
from src.components.data_transformation import FeatureEngineeringTransformer

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from src.utils import load_object

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, val_array):
        try:
            logging.info("splitting to X_train, y_train, X_val, y_val")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val, y_val = val_array[:, :-1], val_array[:, -1]

            preprocessor = load_object("artifacts/data_transformation/preprocessor.pkl")

            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=2),
                "XGB Classifier": XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=2),
                "LGB Classifier": LGBMClassifier(class_weight = 'balanced', random_state=42, n_jobs=2)
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


            report = {}
            logging.info("training in different models and parameters")
            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                para = params[model_name]

                pipeline = Pipeline([
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                # param_grid = params[name]

                logging.info(f"training for {model_name}")

                RS = RandomizedSearchCV(pipeline, param_distributions=para, cv=4, n_iter=10, scoring='roc_auc', n_jobs=6, pre_dispatch=2, verbose=2, random_state=42)
                RS.fit(X_train, y_train)

                best_pipeline = RS.best_estimator_

                y_pred_proba = best_pipeline.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_pred_proba)

                logging.info(f"best parameters for {model_name} are {RS.best_params_} and best score is {roc_auc}")
                report[model_name] = {
                    "model": best_model,
                    "score":roc_auc
                }
            logging.info("training on different models done")
            
            best_model_name = max(report, key=lambda x: report[x]["score"])
            best_model_score = report[best_model_name]["score"]
            best_model = report[best_model_name]["model"]

            logging.info(f"Best model found, model name is {best_model_name}, accuracy score is {best_model_score}")
            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

        except Exception as e:
            raise CustomeException(e, sys)