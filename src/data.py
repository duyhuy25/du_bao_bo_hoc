import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(data_path="data/raw/data.csv"):
    # Always resolve paths relative to the project root
    project_root = os.path.join(os.path.dirname(__file__), "..")
    full_path = os.path.join(project_root, data_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data not found at {full_path}.")
    return pd.read_csv(full_path)

def get_train_val_test_splits(df):
    """
    Split the dataset into Train (60%), Validation (20%), and Test (20%).
    This must be done BEFORE any feature engineering to prevent data leakage.
    """
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split 20% for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split 25% of train_val for validation (0.25 * 0.8 = 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
