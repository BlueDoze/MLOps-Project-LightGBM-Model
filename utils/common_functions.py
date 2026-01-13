import os 
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import joblib

logger = get_logger(__name__)

def read_yaml_file(file_path):
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        logger.info(f"Successfully read YAML file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading YAML file: {file_path}")
        raise CustomException(f"Error reading YAML file: {file_path}") from e


def load_data(file_path):
    """Loads data from a CSV or Excel file into a pandas DataFrame."""
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            data = pd.read_excel(file_path)
        else:
            raise CustomException(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully loaded data from file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from file: {file_path}")
        raise CustomException(f"Error loading data from file: {file_path}") from e


def save_scaler(scaler, file_path):
    """Saves a scaler object to disk using joblib."""
    try:
        joblib.dump(scaler, file_path)
        logger.info(f"Successfully saved scaler to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving scaler to: {file_path}")
        raise CustomException(f"Error saving scaler to: {file_path}") from e


def load_scaler(file_path):
    """Loads a scaler object from disk using joblib."""
    try:
        scaler = joblib.load(file_path)
        logger.info(f"Successfully loaded scaler from: {file_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler from: {file_path}")
        raise CustomException(f"Error loading scaler from: {file_path}") from e