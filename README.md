# 🏨 Hotel Reservation Prediction - MLOps Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-green.svg)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)

A comprehensive MLOps pipeline for predicting hotel reservation booking status using machine learning. This project implements industry-standard MLOps practices including automated data ingestion from Google Cloud Storage, feature engineering, model training with hyperparameter optimization, experiment tracking with MLflow, and comprehensive logging.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Pipeline Components](#pipeline-components)
- [Technologies Used](#technologies-used)
- [Author](#author)

## 🎯 Overview

This project predicts whether a hotel reservation will be confirmed or canceled based on various features such as booking information, customer preferences, and historical patterns. The system uses LightGBM (Light Gradient Boosting Machine) with automated hyperparameter tuning to achieve optimal performance.

**Key Highlights:**
- **Accuracy**: 88.19%
- **Precision**: 91.04%
- **Recall**: 92.08%
- **F1-Score**: 91.56%

## ✨ Features

- **Automated Data Ingestion**: Download datasets directly from Google Cloud Storage
- **Intelligent Preprocessing**: 
  - Automated handling of categorical and numerical features
  - Skewness correction with log transformation
  - SMOTE (Synthetic Minority Over-sampling Technique) for class imbalance
  - Flexible normalization with StandardScaler or MinMaxScaler
  - Feature selection based on importance
- **Advanced Model Training**:
  - LightGBM classifier with RandomizedSearchCV
  - Hyperparameter optimization across multiple dimensions
  - Cross-validation for robust model evaluation
- **Experiment Tracking**: Full MLflow integration for reproducibility
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Custom Exception Handling**: Structured error management with traceback
- **Modular Architecture**: Clean, maintainable, and scalable code structure

## 📁 Project Structure

```
Hotel_Reservation_Prediction/
├── artifacts/              # Generated artifacts
│   ├── evaluation/         # Model evaluation metrics
│   │   └── metrics.json
│   ├── models/             # Trained models
│   │   └── LightGBM_model.pkl
│   ├── params/             # Best hyperparameters
│   │   └── best_params.json
│   ├── processed/          # Processed datasets
│   │   ├── processed_train.csv
│   │   └── processed_test.csv
│   └── raw/                # Raw datasets
│       ├── raw.csv
│       ├── train.csv
│       └── test.csv
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── config.yaml         # Main configuration
│   ├── model_param.py      # Model hyperparameters
│   └── paths_configs.py    # Path configurations
├── logs/                   # Application logs
├── mlruns/                 # MLflow experiment tracking
├── notebook/               # Jupyter notebooks for EDA
│   └── notebook.ipynb
├── pipeline/               # Training pipeline
│   ├── __init__.py
│   └── training_pipeline.py
├── src/                    # Source code
│   ├── __init__.py
│   ├── custom_exception.py # Custom exception handling
│   ├── data_ingestion.py   # Data download and splitting
│   ├── data_preprocessing.py # Feature engineering
│   ├── logger.py           # Logging configuration
│   ├── model_training.py   # Basic model training
│   └── model_training_mlflow.py # MLflow-integrated training
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── common_functions.py
├── requirements.txt        # Project dependencies
├── setup.py                # Package setup
└── git_commit.sh           # Git automation script
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud account (for data ingestion)
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Hotel_Reservation_Prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package and dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set up Google Cloud credentials:**
   - Configure your Google Cloud credentials for accessing GCS buckets
   - Ensure you have access to the bucket specified in the configuration

## ⚙️ Configuration

The project uses a YAML-based configuration system located in [config/config.yaml](config/config.yaml):

### Data Ingestion Configuration
```yaml
data_ingestion:
  bucket_name: "my-bucket1510"
  bucket_file_name: "Hotel_Reservations.csv"
  train_ratio: 0.8
```

### Data Processing Configuration
```yaml
data_processing:
  categorical_columns:
    - type_of_meal_plan
    - required_car_parking_space
    - room_type_reserved
    - market_segment_type
    - repeated_guest
    - booking_status
  
  numerical_columns:
    - no_of_adults
    - no_of_children
    - no_of_weekend_nights
    # ... more features
  
  skewness_threshold: 0.5
  target_column: booking_status
  num_features_to_select: 10
  enable_normalization: true
  scaler_type: "standard"  # Options: "standard" or "minmax"
```

### Model Parameters

Located in [config/model_param.py](config/model_param.py):

```python
LIGHTGBM_PARAM_DISTRIBUTION = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 100),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 4,
    'cv': 5,
    'n_jobs': -1,
    'scoring': 'accuracy'
}
```

## 💻 Usage

### Running the Complete Pipeline

Execute the entire training pipeline:

```bash
python pipeline/training_pipeline.py
```

This will:
1. Download data from Google Cloud Storage
2. Split data into training and testing sets
3. Preprocess and engineer features
4. Train the model with hyperparameter optimization
5. Evaluate model performance
6. Log experiments to MLflow
7. Save artifacts (models, metrics, parameters)

### Running Individual Components

**Data Ingestion:**
```bash
python src/data_ingestion.py
```

**Data Preprocessing:**
```bash
python src/data_preprocessing.py
```

**Model Training:**
```bash
python src/model_training_mlflow.py
```

### Viewing MLflow Experiments

Start the MLflow UI to view experiment tracking:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## 📊 Model Performance

The trained LightGBM model achieves the following performance metrics on the test set:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 88.19% |
| Precision | 91.04% |
| Recall    | 92.08% |
| F1-Score  | 91.56% |

### Best Hyperparameters

```json
{
    "boosting_type": "gbdt",
    "learning_rate": 0.129,
    "max_depth": 23,
    "n_estimators": 314,
    "num_leaves": 94
}
```

## 🔧 Pipeline Components

### 1. Data Ingestion ([src/data_ingestion.py](src/data_ingestion.py))

- Downloads hotel reservation data from Google Cloud Storage
- Splits data into training (80%) and testing (20%) sets
- Stores raw data in the artifacts directory

**Key Methods:**
- `download_data()`: Downloads data from GCS bucket
- `split_data()`: Splits data into train/test sets
- `initiate_data_ingestion()`: Orchestrates the ingestion process

### 2. Data Preprocessing ([src/data_preprocessing.py](src/data_preprocessing.py))

Comprehensive feature engineering pipeline:

- **Data Cleaning**:
  - Removes duplicate records
  - Drops irrelevant columns (Booking_ID)
  
- **Feature Encoding**:
  - Label encoding for categorical variables
  - Maintains consistent encoding between train/test sets
  
- **Skewness Correction**:
  - Log transformation for skewed numerical features
  - Configurable skewness threshold (default: 0.5)
  
- **Class Balancing**:
  - SMOTE oversampling for imbalanced datasets
  - Applied only to training data
  
- **Feature Selection**:
  - Random Forest-based feature importance
  - Selects top N features (configurable)
  
- **Normalization**:
  - StandardScaler or MinMaxScaler options
  - Fit on training data, transform on test data

**Key Methods:**
- `preprocess_data()`: Main preprocessing logic
- `balance_data()`: Applies SMOTE
- `feature_selection()`: Selects important features
- `normalize_features()`: Scales numerical features
- `process()`: Orchestrates the entire preprocessing pipeline

### 3. Model Training ([src/model_training_mlflow.py](src/model_training_mlflow.py))

Advanced training pipeline with experiment tracking:

- **Model**: LightGBM Classifier
- **Optimization**: RandomizedSearchCV
- **Cross-Validation**: 5-fold CV
- **MLflow Integration**:
  - Logs parameters, metrics, and artifacts
  - Tracks experiments for reproducibility
  - Stores models and datasets

**Key Methods:**
- `train_test_split()`: Loads and splits features/targets
- `train_model()`: Trains model with hyperparameter tuning
- `evaluate_model()`: Calculates performance metrics
- `save_evaluated_metrics()`: Persists metrics to JSON
- `save_model()`: Saves trained model as pickle file
- `run()`: Orchestrates training with MLflow tracking

### 4. Utilities ([utils/common_functions.py](utils/common_functions.py))

Reusable helper functions:
- `read_yaml_file()`: Reads YAML configurations
- `load_data()`: Loads CSV/Excel files into DataFrames
- `save_scaler()`: Persists scikit-learn scalers
- `load_scaler()`: Loads saved scalers

### 5. Logging & Exception Handling

**Logger ([src/logger.py](src/logger.py)):**
- Timestamp-based log files
- INFO level logging
- Detailed execution tracking

**Custom Exceptions ([src/custom_exception.py](src/custom_exception.py)):**
- Structured error messages
- Full traceback information
- Consistent error handling across modules

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | LightGBM, scikit-learn, imbalanced-learn |
| **Data Processing** | pandas, numpy |
| **Experiment Tracking** | MLflow |
| **Cloud Storage** | Google Cloud Storage |
| **Visualization** | matplotlib, seaborn |
| **Configuration** | PyYAML |
| **Development** | Jupyter Notebook, ipykernel |

### Key Dependencies

```
pandas
numpy
google-cloud-storage
scikit-learn
pyyaml
ipykernel
matplotlib
seaborn
imbalanced-learn
lightgbm
mlflow
```

## 📈 Future Enhancements

- [ ] API deployment with FastAPI/Flask
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model serving with MLflow Model Registry
- [ ] Feature store integration
- [ ] Real-time prediction endpoint
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Advanced feature engineering
- [ ] Ensemble modeling

## 👤 Author

**Da Silva Aguiar, Pedro**

## 📝 License

This project is available for educational and portfolio purposes.

---

**Note**: Ensure you have proper Google Cloud credentials configured before running the data ingestion pipeline. Update the `config/config.yaml` file with your GCS bucket details.

For questions or suggestions, feel free to open an issue or submit a pull request!
