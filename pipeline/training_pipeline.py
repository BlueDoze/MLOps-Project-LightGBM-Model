from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training_mlflow import ModelTrainer
from utils.common_functions import *
from src.logger import get_logger
from config.paths_configs import *
from config.model_param import *

logger = get_logger(__name__)

if __name__ == "__main__":

    logger.info("Starting Training Pipeline...")
    logger.info("Starting Data Ingestion...")
    ## 1. Data Ingestion
    config = read_yaml_file("config/config.yaml")
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion()
    logger.info("Data Ingestion Completed.")

    logger.info("Starting Data Preprocessing...")
    ## 2. Data Preprocessing
    data_preprocessing = DataPreprocessing(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_preprocessing.process()
    logger.info("Data Preprocessing Completed.")

    logger.info("Starting Model Training with MLflow...")
    ## 3. Model TRaining Mlflow
    model_trainer = ModelTrainer(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    model_trainer.run()
    logger.info("Model Training Completed.")