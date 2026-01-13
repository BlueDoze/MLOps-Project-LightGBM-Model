import lightgbm as lgb
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.paths_configs import *
from config.model_param import *
from sklearn.model_selection import RandomizedSearchCV
from utils.common_functions import read_yaml_file, load_data
from src.logger import get_logger
from src.custom_exception import CustomException
import json

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, train_path: str, test_path: str, model_output_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.best_params_path = BEST_PARAMS_PATH
        self.config_target = read_yaml_file(CONFIG_PATH)['data_processing']['target_column']

        logger.info("ModelTrainer initialized with train path: %s, test path: %s, model output path: %s", train_path, test_path, model_output_path)
    
    def train_test_split(self):
        
        logger.info("Loading training and testing data...")
        train_data = load_data(self.train_path)
        test_data = load_data(self.test_path)

        logger.info("Splitting features and target variable")
        X_train = train_data.drop(self.config_target, axis=1)
        y_train = train_data[self.config_target]
        X_test = test_data.drop(self.config_target, axis=1)
        y_test = test_data[self.config_target]

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train, X_test):
        logger.info("Starting model training process...")

        logger.info("Initializing LightGBM model")
        lgb_model = lgb.LGBMClassifier()

        logger.info("Setting up RandomizedSearchCV")
        random_search = RandomizedSearchCV(
            estimator=lgb_model,
            param_distributions=LIGHTGBM_PARAM_DISTRIBUTION,
            n_iter=RANDOM_SEARCH_PARAMS['n_iter'],
            cv=RANDOM_SEARCH_PARAMS['cv'],
            n_jobs=RANDOM_SEARCH_PARAMS['n_jobs'],
            verbose=RANDOM_SEARCH_PARAMS['verbose'],
            random_state=RANDOM_SEARCH_PARAMS['random_state'],
            scoring=RANDOM_SEARCH_PARAMS['scoring']
        )

        logger.info("Fitting model with RandomizedSearchCV")
        random_search.fit(X_train, y_train)

        logger.info("Best parameters found: %s", random_search.best_params_)
        best_parameters = random_search.best_params_
        best_model = random_search.best_estimator_


        logger.info("Making predictions on test data")
        y_pred = best_model.predict(X_test)

        return y_pred, best_model, best_parameters

    def evaluate_model(self, y_test, y_pred):

        logger.info("Evaluating model performance")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return accuracy, precision, recall, f1

    def save_evaluated_metrics(self, accuracy, precision, recall, f1, best_parameters, output_path="artifacts/evaluation/metrics.json"):

        logger.info("Saving evaluation metrics")
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        os.mkdir(os.path.dirname(output_path)) if not os.path.exists(os.path.dirname(output_path)) else None

        logger.info("Creating metrics file at %s", output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation metrics saved at %s", output_path)

        logger.info("Saving the best parameters...")
        os.mkdir(os.path.dirname(self.best_params_path)) if not os.path.exists(os.path.dirname(self.best_params_path)) else None
        with open(self.best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_parameters, f, indent=4)
        logger.info("Best parameters saved at %s", self.best_params_path)

    def save_model(self, best_model):
        os.mkdir(os.path.dirname(self.model_output_path)) if not os.path.exists(os.path.dirname(self.model_output_path)) else None
        joblib.dump(best_model, self.model_output_path)
        logger.info("Model saved at %s", self.model_output_path)

if __name__ == "__main__":
    try:
        logger.info("Starting model training process...")
        model_trainer = ModelTrainer(
            train_path=PROCESSED_TRAIN_DATA_PATH,
            test_path=PROCESSED_TEST_DATA_PATH,
            model_output_path=MODEL_OUTPUT_PATH
        )

        logger.info("Splitting data into train and test sets...")
        X_train, y_train, X_test, y_test = model_trainer.train_test_split()

        logger.info("Training model...")
        y_pred, best_model, best_parameters = model_trainer.train_model(X_train, y_train, X_test)

        logger.info("Evaluating model...")
        accuracy, precision, recall, f1 = model_trainer.evaluate_model(y_test, y_pred)

        logger.info("Model Evaluation Metrics - Accuracy: %s, Precision: %s, Recall: %s, F1 Score: %s", accuracy, precision, recall, f1)
        model_trainer.save_evaluated_metrics(accuracy, precision, recall, f1, best_parameters)

        logger.info("Saving trained model...")
        model_trainer.save_model(best_model)

        logger.info("Model training process completed successfully")

    except Exception as e:
        logger.error("An error occurred during model training: %s", e)
        raise CustomException(e) from e