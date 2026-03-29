import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object
from src.components.data_transformation import DataTransformation

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
    preprocessor_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_val, y_train, y_val):
        try:
            logging.info("splitting to X_train, y_train, X_val, y_val")
            # X_train, y_train = train_array[:, :-1], train_array[:, -1]
            # X_val, y_val = val_array[:, :-1], val_array[:, -1]

            # data_transformation = DataTransformation()
            preprocessor = load_object(self.model_trainer_config.preprocessor_file_path)

            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                "XGB Classifier": XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=2),
                "LGB Classifier": LGBMClassifier(class_weight = 'balanced', random_state=42, n_jobs=2)
            }

            params = {
                "Random Forest":{
                    'model__n_estimators':[100],
                    'model__max_features': ['sqrt'],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5]
                },
                "XGB Classifier": {
                    'model__n_estimators': [100],
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [6, 10],
                    'model__subsample': [0.8, 1.0],
                    'model__colsample_bytree': [0.8, 1.0],
                    'model__scale_pos_weight': [1, 5, 10]
                },
                "LGB Classifier": {
                    'model__n_estimators': [100, 300],
                    'model__learning_rate': [0.05, 0.1],
                    'model__num_leaves': [31, 64],
                    'model__max_depth': [-1, 10],
                    'model__subsample': [0.8, 1.0],
                    'model__colsample_bytree': [0.8, 1.0],
                    'model__scale_pos_weight': [1, 5, 10]   # IMPORTANT for imbalance
                }
            }


            report = {}
            logging.info("training in different models and parameters")
            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                # para = params[model_name]

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                # param_grid = params[name]

                logging.info(f"training for {model_name}")

                RS = RandomizedSearchCV(pipeline, param_distributions=params[model_name], cv=4, n_iter=8, scoring='roc_auc', n_jobs=4, pre_dispatch=2, verbose=2, random_state=42)
                RS.fit(X_train, y_train)

                best_pipeline = RS.best_estimator_

                y_pred_proba = best_pipeline.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_pred_proba)

                logging.info(f"{model_name} best params: {RS.best_params_}")
                logging.info(f"{model_name} ROC-AUC: {roc_auc}")
                report[model_name] = {
                    "model": best_pipeline,
                    "score":roc_auc
                }
            logging.info("training on different models done")
            
            best_model_name = max(report, key=lambda x: report[x]["score"])
            best_model_score = report[best_model_name]["score"]
            best_model = report[best_model_name]["model"]

            logging.info(f"Best model found, model name is {best_model_name}, accuracy score is {best_model_score}")
            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomeException(e, sys)