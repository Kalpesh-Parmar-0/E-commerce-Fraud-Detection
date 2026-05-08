import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
    feature_maps_path = os.path.join("artifacts/data_transformation", "feature_maps.pkl")
    columns_path = os.path.join("artifacts/data_transformation", "columns.pkl")

class FeatureMaps:
    def __init__(self):
        self.freq_maps = {}
        self.count_maps = {}
        self.agg_maps = {}

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

            new_cols = {}

            # D column transformations (from Kaggle notebooks)
            d_cols = [col for col in df.columns if col.startswith("D")]

            for col in d_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    new_cols[col + "_log"] = np.log1p(df[col].fillna(0).clip(lower=0))
                else:
                    logging.warning(f"Skipping log transformation for non-numeric column {col}")

            # Transaction amount transformation
            if "TransactionAmt" in df.columns:
                if pd.api.types.is_numeric_dtype(df["TransactionAmt"]):
                    new_cols["TransactionAmt_log"] = np.log1p(
                    df["TransactionAmt"].fillna(0).clip(lower=0)
                    )
                else:
                    logging.warning("Skipping log transformation for TransactionAmt as it's not numeric")

            # Email match feature
            if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
                new_cols["email_match"] = (
                    df["P_emaildomain"] == df["R_emaildomain"]
                ).astype(int)

            # UID features

            if {'card1', 'card2'}.issubset(df.columns):
                uid = df['card1'].astype(str) + "_" + df['card2'].astype(str)
                new_cols['uid'] = uid

            if 'uid' in new_cols and 'addr1' in df.columns:
                uid2 = uid + "_" + df['addr1'].astype(str)
                new_cols['uid2'] = uid2

            if 'uid2' in new_cols and 'P_emaildomain' in df.columns:
                new_cols['uid3'] = uid2 + "_" + df['P_emaildomain'].astype(str)

            if new_cols:
                df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
            df = df.copy() 

            logging.info("Feature engineering completed")
            return df
        
        except Exception as e:
            raise CustomeException(e, sys)
        
    # transaction aggregation
    def transaction_aggregations(self,train_df, test_df, maps: FeatureMaps):
        agg_cols = ["card1", "card2", "addr1", "uid"]

        for col in agg_cols:
            if col in train_df.columns and "TransactionAmt" in train_df.columns:
                agg = train_df.groupby(col)['TransactionAmt'].agg(['mean','std']).astype('float32')
                agg.columns = [f"{col}_amt_mean", f"{col}_amt_std"]

                maps.agg_maps[col] = agg

                train_df = train_df.merge(agg, on=col, how="left")
                test_df = test_df.merge(agg, on=col, how="left")

                test_df[f"{col}_amt_mean"].fillna(-1, inplace=True)
                test_df[f"{col}_amt_std"].fillna(-1, inplace=True)

                train_df[f"{col}_amt_diff"] = train_df["TransactionAmt"] - train_df[f"{col}_amt_mean"]
                test_df[f"{col}_amt_diff"] = test_df["TransactionAmt"] - test_df[f"{col}_amt_mean"]
        return train_df, test_df
    
    def apply_transaction_aggregations(self, df, maps: FeatureMaps):
        for col, agg in maps.agg_maps.items():
            df = df.merge(agg, on=col, how="left")

            df[f"{col}_amt_diff"] = df["TransactionAmt"] - df[f"{col}_amt_mean"]

            df[f"{col}_amt_mean"].fillna(-1, inplace=True)
            df[f"{col}_amt_std"].fillna(-1, inplace=True)

        return df
    
    # transacton counts
    def transaction_counts(self, train_df, test_df, maps: FeatureMaps):
        cols = ["card1", "uid"]

        for col in cols:
            if col in train_df.columns:
                count = train_df[col].value_counts()

                maps.count_maps[col] = count

                train_df[col + "_count"] = train_df[col].map(count)
                test_df[col + "_count"] = test_df[col].map(count)

        return train_df, test_df
        
    # FREQUENCY ENCODING
    def frequency_encoding(self, train_df, test_df, columns, maps: FeatureMaps):

        for col in columns:
            if col in train_df.columns:
                freq = train_df[col].value_counts(normalize=True)

                maps.freq_maps[col] = freq

                train_df[col] = train_df[col].map(freq)
                test_df[col] = test_df[col].map(freq)

                test_df[col].fillna(0, inplace=True)
        return train_df, test_df
    
    def reduce_memory(self, df):
        try:
            logging.info("Reducing memory usage of dataframe")
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    df[col] = df[col].astype('int32')
            return df
        except Exception as e:
            raise CustomeException(e, sys)

    # preprocessing pipeline
    def get_preprocessor(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        
        
    # run complete transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('loading dataset')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data loaded")

            maps = FeatureMaps()

            # column cleaning
            train_df = self.clean_columns(train_df)
            test_df = self.clean_columns(test_df)

            # feature enginerring
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            train_df, test_df = self.transaction_aggregations(train_df, test_df, maps)
            train_df, test_df = self.transaction_counts(train_df, test_df, maps)

            # reduce memory
            train_df = self.reduce_memory(train_df)
            test_df = self.reduce_memory(test_df)

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
                cat_cols,
                maps
            )

            target_column = "isFraud"

            X = train_df.drop(columns=[target_column])
            X = X.select_dtypes(exclude=["object"])
            y = train_df[target_column]

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # preprocessor
            preprocessor = self.get_preprocessor()
            preprocessor.fit(X_train)

            save_object(self.data_transformation_config.preprocess_obj_file_path, preprocessor)
            save_object(self.data_transformation_config.feature_maps_path, maps)
            save_object(self.data_transformation_config.columns_path, X_train.columns.tolist())

            # ✅ FREE MEMORY HERE
            del train_df, test_df, X, y
            import gc
            gc.collect()

            logging.info('preprosessor saved')

            return (
                X_train, X_val, y_train, y_val
            )

        except Exception as e:
            raise CustomeException(e, sys)
        
# class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.data_transformation = DataTransformation()

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = self.data_transformation.clean_columns(X)
#         X = self.data_transformation.feature_engineering(X)

#         # ⚠️ keep only numeric
#         X = X.select_dtypes(exclude=["object"])

#         return X