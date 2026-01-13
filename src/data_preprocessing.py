import pandas as pd
import numpy as np
import os
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_configs import *
from utils.common_functions import *
from utils.common_functions import save_scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml_file(config_path)

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        logger.info("DataPreprocessing instance created.")

    def preprocess_data(self, df, label_encoders=None, is_training=True):
        """Preprocesses the data by handling missing values, encoding categorical variables, and balancing classes.
        
        Args:
            df: Input dataframe to preprocess
            label_encoders: Dictionary of fitted label encoders (for test data)
            is_training: Boolean indicating if this is training data
            
        Returns:
            Tuple of (preprocessed_df, label_encoders)
        """
        try:
            logger.info("Starting data preprocessing.")

            logger.info("Droping Columns.")
            df.drop(columns=['Booking_ID'] , inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']


            logger.info("Columns Dropped Successfully.")

    
            # Encoding categorical variables
            if is_training:
                # Fit encoders on training data
                label_encoders = {}
                mappings = {}
                for column in cat_cols:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    label_encoders[column] = le
                    mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
                logger.info("Categorical variables encoded successfully.")

                logger.info("Label Mapping are:")
                for col, mapping in mappings.items():
                    logger.info(f"{col}: {mapping}")
            else:
                # Transform test data using fitted encoders from training
                if label_encoders is None:
                    raise CustomException("Label encoders must be provided for test data preprocessing.")
                for column in cat_cols:
                    # Handle unseen categories by mapping them to a default value (most frequent class)
                    df[column] = df[column].map(lambda x: x if x in label_encoders[column].classes_ else label_encoders[column].classes_[0])
                    df[column] = label_encoders[column].transform(df[column])
                logger.info("Test data categorical variables transformed using training encoders.")
            
            logger.info("Handling skewed data.")

            skewness_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > skewness_threshold].index:
                df[col] = np.log1p(df[col])
            logger.info("Skewed data handled successfully.")

            logger.info("Data preprocessing completed successfully.")

            return df, label_encoders
        
        except Exception as e:
            logger.error("Error during data preprocessing.")
            raise CustomException("Error during data preprocessing.") from e
    
    def balance_data(self, df):
        """Balances the dataset using SMOTE."""
        try:
            logger.info("Handling imbalanced data using SMOTE.")
            X = df.drop(columns=[self.config['data_processing']['target_column']])
            y = df[self.config['data_processing']['target_column']]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df[self.config['data_processing']['target_column']] = y_resampled

            logger.info("Data balanced successfully using SMOTE.")

            return balanced_df
        
        except Exception as e:
            logger.error("Error during data balancing.")
            raise CustomException("Error during data balancing.") from e
    
    def select_features(self, df):
        """Selects important features using RandomForestClassifier."""
        try:
            logger.info("Starting feature selection using RandomForestClassifier.")

            X = df.drop(columns=[self.config['data_processing']['target_column']])
            y = df[self.config['data_processing']['target_column']]

            rf = RandomForestClassifier(random_state=42)
            rf.fit(X, y)

            feature_importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({'feature':X.columns, 'importance':feature_importances}).sort_values(by='importance', ascending=False)
            
            num_features_to_select = self.config['data_processing']['num_features_to_select']

            top_10_features = feature_importance_df["feature"].head(num_features_to_select).values

            top_10_df = df[top_10_features.tolist() + [self.config['data_processing']['target_column']]]

            selected_df = top_10_df.copy()

            logger.info("Feature selection completed successfully.")

            return selected_df
        
        except Exception as e:
            logger.error("Error during feature selection.")
            raise CustomException("Error during feature selection.") from e
    
    def normalize_data(self, train_df, test_df):
        """Normalizes numerical features using StandardScaler or MinMaxScaler.
        Fits scaler on training data only to prevent data leakage.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of (normalized_train_df, normalized_test_df, fitted_scaler)
        """
        try:
            # Check if normalization is enabled
            if not self.config['data_processing'].get('enable_normalization', False):
                logger.info("Normalization is disabled. Skipping normalization step.")
                return train_df, test_df, None
            
            logger.info("Starting data normalization.")
            
            # Get numerical columns
            num_cols = self.config['data_processing']['numerical_columns']
            target_col = self.config['data_processing']['target_column']
            
            # Filter numerical columns that exist in the dataframes (after feature selection)
            num_cols_to_scale = [col for col in num_cols if col in train_df.columns and col != target_col]
            
            if not num_cols_to_scale:
                logger.info("No numerical columns to normalize.")
                return train_df, test_df, None
            
            # Select scaler type
            scaler_type = self.config['data_processing'].get('scaler_type', 'standard')
            if scaler_type == 'standard':
                scaler = StandardScaler()
                logger.info("Using StandardScaler for normalization.")
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
                logger.info("Using MinMaxScaler for normalization.")
            else:
                raise CustomException(f"Invalid scaler type: {scaler_type}. Use 'standard' or 'minmax'.")
            
            # Fit scaler on training data only (CRITICAL: prevents data leakage)
            train_df_copy = train_df.copy()
            test_df_copy = test_df.copy()
            
            scaler.fit(train_df[num_cols_to_scale])
            logger.info(f"Scaler fitted on training data with {len(num_cols_to_scale)} numerical columns.")
            
            # Transform both training and test data using the fitted scaler
            train_df_copy[num_cols_to_scale] = scaler.transform(train_df[num_cols_to_scale])
            test_df_copy[num_cols_to_scale] = scaler.transform(test_df[num_cols_to_scale])
            
            logger.info("Data normalization completed successfully.")
            logger.info(f"Normalized columns: {num_cols_to_scale}")
            
            return train_df_copy, test_df_copy, scaler
        
        except Exception as e:
            logger.error("Error during data normalization.")
            raise CustomException("Error during data normalization.") from e
            
    def save_data(self, df, file_path):
        """Saves the DataFrame to a CSV file."""
        try:
            logger.info(f"Saving data to {file_path}.")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}.")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}.")
            raise CustomException(f"Error saving data to {file_path}.") from e
        
    def process(self):
        """Main method to process the data."""
        try:
            logger.info("Initiating data processing.")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # Preprocess training data (fit encoders)
            train_df, label_encoders = self.preprocess_data(train_df, is_training=True)
            # Preprocess test data (use fitted encoders)
            test_df, _ = self.preprocess_data(test_df, label_encoders=label_encoders, is_training=False)

            # Normalize data (fit scaler on train, transform both)
            train_df, test_df, scaler = self.normalize_data(train_df, test_df)
            
            # Save the fitted scaler for future use
            if scaler is not None:
                scaler_path = os.path.join(self.processed_dir, 'scaler.pkl')
                save_scaler(scaler, scaler_path)
                logger.info(f"Scaler saved to: {scaler_path}")

            train_df = self.balance_data(train_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully.")

        except Exception as e:
            logger.error("Data processing failed.")
            raise CustomException("Data processing failed.") from e


if __name__ == "__main__":
    try:
        data_preprocessing = DataPreprocessing(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH
        )
        data_preprocessing.process()
    except Exception as e:
        logger.error("Data preprocessing process failed.")
        raise CustomException("Data preprocessing process failed.") from e