import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_configs import RAW_DIR, RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
from utils.common_functions import read_yaml_file

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config_path):
        self.config = config_path["data_ingestion"]
        self.bucket_name = self.config['bucket_name']
        self.bucket_file_name = self.config['bucket_file_name']
        self.train_ratio = self.config['train_ratio']

        os.mkdir(RAW_DIR) if not os.path.exists(RAW_DIR) else None

        logger.info("DataIngestion instance created with configuration.")

    def download_data(self):
        """Downloads data from Google Cloud Storage."""
        try:
            logger.info(f"Starting data download from GCS bucket: {self.bucket_name}, file: {self.bucket_file_name}")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Data downloaded successfully from {self.bucket_name}/{self.bucket_file_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error("Failed to download data from GCS.")
            raise CustomException("Failed to download data from GCS.") from e

    def split_data(self):
        """Splits the raw data into training and testing sets."""
        try:
            logger.info("Starting data split into train and test sets.")
            df = pd.read_csv(RAW_FILE_PATH)
            train_df, test_df = train_test_split(df, train_size=self.train_ratio, random_state=42)
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Data split into train and test sets with ratio {self.train_ratio}.")
        except Exception as e:
            logger.error("Failed to split data into train and test sets.")
            raise CustomException("Failed to split data into train and test sets.") from e

    def initiate_data_ingestion(self):
        """Initiates the data ingestion process."""
        logger.info("Initiating data ingestion process.")
        self.download_data()
        self.split_data()
        logger.info("Data ingestion process completed successfully.")

if __name__ == "__main__":
    try:
        config = read_yaml_file("config/config.yaml")
        data_ingestion = DataIngestion(config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logger.error("Data ingestion process failed.")
        raise CustomException("Data ingestion process failed.") from e