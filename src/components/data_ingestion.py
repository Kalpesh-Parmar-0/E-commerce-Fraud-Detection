import os, sys
from src.logger import logging
from src.exception import CustomeException
from dataclasses import dataclass

import pandas as pd

@dataclass
class DataIngestionConfig:
    raw_train_transactions: str = os.path.join('data', 'raw', 'train_transaction.csv')
    raw_train_identity: str = os.path.join('data', 'raw', 'train_identity.csv')

    raw_test_transactions: str = os.path.join('data', 'raw', 'test_transaction.csv')
    raw_test_identity: str = os.path.join('data', 'raw', 'test_identity.csv')

    merged_train_path = os.path.join("artifacts", "train_merged.csv")
    merged_test_path = os.path.join("artifacts", "test_merged.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('data ingestion started')
        try:
            logging.info('train data reading using pandas')
            train_trans = pd.read_csv(self.ingestion_config.raw_train_transactions)
            train_id = pd.read_csv(self.ingestion_config.raw_train_identity)
            logging.info('train data reading complete')

            logging.info('test data reading start using pandas')
            test_trans = pd.read_csv(self.ingestion_config.raw_test_transactions)
            test_id = pd.read_csv(self.ingestion_config.raw_test_identity)
            logging.info('test data reading complete')

            logging.info("transaction and identity merge start")
            train_df = train_trans.merge(train_id, on="TransactionID", how="left")
            test_df = test_trans.merge(test_id, on="TransactionID", how="left")
            logging.info('transaction and identity merged in train_df and test_df')

            os.makedirs("artifacts", exist_ok=True)
            train_df.to_csv(self.ingestion_config.merged_train_path, index=False)
            test_df.to_csv(self.ingestion_config.merged_test_path, index=False)
            
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.merged_train_path,
                self.ingestion_config.merged_test_path
            )

        except Exception as e:
            logging.info('Error accoured at data ingestion stage')
            raise CustomeException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
