import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    # clean columns
    def clean_columns(self, df):
        try:
            logging.info('cleaning columns')

            id_cols = ['TransactionID', 'TransactionDT']
            reduced_vcols = [
                'V1','V3','V4','V6','V8','V11','V13','V14','V17','V20',
                'V23','V26','V27','V30','V36','V37','V40','V41','V44',
                'V47','V48','V54','V56','V59','V62','V65','V67','V68',
                'V70','V76','V78','V80','V82','V86','V88','V89','V91',
                'V96','V98','V99','V104','V107','V108','V111','V115',
                'V117','V120','V121','V123','V124','V127','V129','V130',
                'V136','V138','V139','V142','V147','V156','V160','V162',
                'V165','V166','V169','V171','V173','V175','V176','V178',
                'V180','V182','V185','V187','V188','V198','V203','V205',
                'V207','V209','V210','V215','V218','V220','V221','V223',
                'V224','V226','V228','V229','V234','V235','V238','V240',
                'V250','V252','V253','V257','V258','V260','V261','V264',
                'V266','V267','V271','V274','V277','V281','V283','V284',
                'V285','V286','V289','V291','V294','V296','V297','V301',
                'V303','V305','V307','V309','V310','V314','V320','V325',
                'V332','V335','V338'
            ]
            drop_v_cols = [col for col in df.columns if col.startswith("V") and col not in reduced_vcols]

            df = df.drop(columns=drop_v_cols, errors="ignore")
            df = df.drop(columns=id_cols, errors="ignore")

            logging.info(f"Remaining columns: {len(df.columns)}")
            return df

        except Exception as e:
            raise CustomeException(e, sys)
        
    # feature engineering
    def feature_engineering(self, df):
        try:
            logging.info("Starting feature engineering")

            # D column transformations (from Kaggle notebooks)
            d_cols = [col for col in df.columns if col.startswith("D")]

            for col in d_cols:
                df[col + "_log"] = np.log1p(df[col])

            # Transaction amount transformation
            if "TransactionAmt" in df.columns:
                df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

            # Email match feature
            if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
                df["email_match"] = (
                    df["P_emaildomain"] == df["R_emaildomain"]
                ).astype(int)

            logging.info("Feature engineering completed")
            return df
        
        except Exception as e:
            raise CustomeException(e, sys)
        
    # FREQUENCY ENCODING
    def frequency_encoding(self, train_df, test_df, columns):

        for col in columns:
            freq = train_df[col].value_counts(normalize=True)

            train_df[col] = train_df[col].map(freq)
            test_df[col] = test_df[col].map(freq)

        return train_df, test_df

    # preprocessing pipeline
    def get_data_transformation_object(self, df):
        try:
            logging.info('Creating preprocessing pipeline')
            target_column = 'isFraud'

            cat_cols = (['ProductCD'] + 
            ['card%d' % i for i in range(1, 7)] + 
            ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + 
            ['M%d' % i for i in range(1, 10)] + 
            ['DeviceType', 'DeviceInfo'] +
            ['id_%d' % i for i in range(12, 39)])

            numerical_columns = df.drop(columns=[target_column] + cat_cols, errors="ignore").columns.tolist()
            categorical_columns = [col for col in cat_cols if col in df.columns]

            logging.info(f"Numerical columns count: {len(numerical_columns)}")
            logging.info(f"Categorical columns count: {len(categorical_columns)}")

            # numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # categorical pipeline
            # cat_pipeline = Pipeline(
            #     steps=[
            #         ("imputer", SimpleImputer(strategy="most_frequent")),
            #         ("onehot", OneHotEncoder(handle_unknown="ignore"))
            #     ]
            # )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    # ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomeException(e, sys)
        
    # run complete transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('loading dataset')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data loaded")

            # column cleaning
            train_df = self.clean_columns(train_df)
            test_df = self.clean_columns(test_df)

            # feature enginerring
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            cat_cols = (
                ['ProductCD'] +
                ['card%d' % i for i in range(1, 7)] +
                ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] +
                ['M%d' % i for i in range(1, 10)] +
                ['DeviceType', 'DeviceInfo'] +
                ['id_%d' % i for i in range(12, 39)]
            )

            train_df, test_df = self.frequency_encoding(
                train_df,
                test_df,
                cat_cols
            )

            preprocess_obj = self.get_data_transformation_object(train_df)

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