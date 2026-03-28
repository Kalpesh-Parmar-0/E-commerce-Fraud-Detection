from src.logger import logging
from src.exception import CustomeException
import os, sys
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        logging.info("training in different models and parameters")
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            logging.info(f"training for {model_name}")

            RS = RandomizedSearchCV(model, param_distributions=para, cv=4, n_iter=15, scoring='roc_auc', n_jobs=6, pre_dispatch=2, verbose=2, random_state=42)
            RS.fit(X_train, y_train)

            best_model = RS.best_estimator_

            logging.info(f"best parameters for {model_name} are {RS.best_params_}")

            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            report[model_name] = {
                "model": best_model,
                "score":roc_auc
            }
        logging.info("training on different models done")
        return report

    except Exception as e:
        raise CustomeException(e, sys)
