import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.data import load_data, get_train_val_test_splits

def evaluate_model():
    print("Loading data...")
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits(df)
    
    project_root = os.path.join(os.path.dirname(__file__), "..")
    model_path = os.path.join(project_root, "models", "dropout_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")
        
    print("Loading model...")
    pipeline = joblib.load(model_path)
    
    print("\n--- Evaluation on Validation Set ---")
    y_val_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_val_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred, labels=pipeline.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pipeline.classes_, 
                yticklabels=pipeline.classes_)
    plt.title('Confusion Matrix on Validation Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "confusion_matrix.png")
    
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"\nConfusion matrix plot saved to {fig_path}")

if __name__ == "__main__":
    evaluate_model()
