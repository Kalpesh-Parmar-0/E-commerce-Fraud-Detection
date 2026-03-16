import os, sys
from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Creating preprocessing pipeline')

        except Exception as e:
            raise CustomeException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded")

            preprocess_obj = self.get_data_transformation_object()

            target_column = "isFraud"

            logging.info("Splitting train data into dependent and independent features")
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            logging.info("Splitting test data into dependent and independent features")
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Appling preprocessing object on training datafrem and testing dataframe")
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_df)
            input_test_arr = preprocess_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test_df)]

            save_object (
                file_path = self.data_transformation_config.preprocess_obj_file_path,
                obj = preprocess_obj
            )

            logging.info('preprosessor saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )

        except Exception as e:
            raise CustomeException(e, sys)