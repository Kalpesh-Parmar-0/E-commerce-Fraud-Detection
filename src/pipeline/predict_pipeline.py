import os, sys
from src.logger import logging
import pandas as pd
from src.exception import CustomeException
from dataclasses import dataclass
from src.utils import load_object
from src.components.data_transformation import DataTransformation, FeatureMaps

class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object("artifacts/model_trainer/model.pkl")
            self.maps = load_object("artifacts/data_transformation/feature_maps.pkl")
            self.columns = load_object("artifacts/data_transformation/columns.pkl")
            self.preprocessor = load_object("artifacts/data_transformation/preprocessor.pkl")

            self.data_transformation = DataTransformation()

            logging.info("Prediction pipeline loaded successfully")

        except Exception as e:
            raise CustomeException(e, sys)

    def apply_transaction_aggregations(self, df):
        for col, agg in self.maps.agg_maps.items():
            if col in df.columns:
                df = df.merge(agg, on=col, how="left")

                df[f"{col}_amt_diff"] = df["TransactionAmt"] - df[f"{col}_amt_mean"]

                df[f"{col}_amt_mean"].fillna(-1, inplace=True)
                df[f"{col}_amt_std"].fillna(-1, inplace=True)

        return df
    
    def apply_transaction_counts(self, df):
        for col, count_map in self.maps.count_maps.items():
            if col in df.columns:
                df[col + "_count"] = df[col].map(count_map).fillna(0)
        return df
    
    # ✅ apply frequency encoding
    def apply_frequency_encoding(self, df):
        for col, freq_map in self.maps.freq_maps.items():
            if col in df.columns:
                df[col] = df[col].map(freq_map)
                df[col].fillna(0, inplace=True)
        return df

    def predict(self, features:pd.DataFrame):
        try:
            logging.info("Starting prediction")
            df = self.data_transformation.clean_columns(features)
            df = self.data_transformation.feature_engineering(df)

            df = self.apply_transaction_aggregations(df)
            df = self.apply_transaction_counts(df)
            df = self.apply_frequency_encoding(df)

            # keep only numeric
            df = df.select_dtypes(exclude=["object"])

            # ensure same columns as training — fill missing with training medians, not 0
            median_defaults = pd.Series(self.preprocessor.statistics_, index=self.columns)
            df = df.reindex(columns=self.columns)

            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(median_defaults[col])

            df = self.data_transformation.reduce_memory(df)
            
            proba = self.model.predict_proba(df)[:, 1]
            logging.info(f"Fraud probability: {proba[0]}")

            # custom threshold (very important for imbalanced data)
            threshold = 0.2

            preds = (proba > threshold).astype(int)
            
            return preds, proba
        
        except Exception as e:
            raise CustomeException(e, sys)
        
class CustomClass:
    def __init__(
            self, 
            TransactionAmt: float, 
            ProductCD: str, 
            card1: int, 
            card2: float, 
            card3: float, 
            card4: str, 
            card5: float, 
            card6: str, 
            addr1: float, 
            addr2: float, 
            P_emaildomain: str, 
            R_emaildomain: str, 
            DeviceType: str, 
            DeviceInfo: str
        ):
        self.TransactionAmt = TransactionAmt
        self.ProductCD = ProductCD
        self.card1 = card1
        self.card2 = card2
        self.card3 = card3
        self.card4 = card4
        self.card5 = card5
        self.card6 = card6
        self.addr1 = addr1
        self.addr2 = addr2
        self.P_emaildomain = P_emaildomain
        self.R_emaildomain = R_emaildomain
        self.DeviceType = DeviceType
        self.DeviceInfo = DeviceInfo
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "TransactionAmt": [self.TransactionAmt],
                "ProductCD": [self.ProductCD],
                "card1": [self.card1],
                "card2": [self.card2],
                "card3": [self.card3],
                "card4": [self.card4],
                "card5": [self.card5],
                "card6": [self.card6],
                "addr1": [self.addr1],
                "addr2": [self.addr2],
                "P_emaildomain": [self.P_emaildomain],
                "R_emaildomain": [self.R_emaildomain],
                "DeviceType": [self.DeviceType],
                "DeviceInfo": [self.DeviceInfo],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomeException(e, sys)