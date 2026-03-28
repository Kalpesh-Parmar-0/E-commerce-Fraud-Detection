import os, sys
from src.logger import logging
from src.exception import CustomeException
from dataclasses import dataclass
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.pipeline = load_object("artifacts/model_trainer/model.pkl")

    def predict(self, features):
        try:
            return self.pipeline.predict(features)
        
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