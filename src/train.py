import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.data import load_data, get_train_val_test_splits
from src.features import DropFutureFeatures, build_preprocessor

def train_model():
    print("Loading data...")
    df = load_data()
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits(df)
    
    print("Building pipeline...")
    preprocessor = build_preprocessor(X_train)
    
    pipeline = Pipeline(steps=[
        ('dropper', DropFutureFeatures()),
        ('preprocessor', preprocessor),
        # Using class_weight='balanced' to improve Recall for minority classes like Dropout
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    print("Training model on Train set...")
    pipeline.fit(X_train, y_train)
    
    # Save model
    project_root = os.path.join(os.path.dirname(__file__), "..")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dropout_model.pkl")
    
    joblib.dump(pipeline, model_path)
    print("Model saved successfully to models/dropout_model.pkl")

if __name__ == "__main__":
    train_model()
