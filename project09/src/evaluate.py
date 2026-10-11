import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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

    # Đánh giá theo nhóm (Subgroup Evaluation)
    print("\n--- Subgroup Evaluation on Validation Set ---")
    subgroup_cols = ['Gender', 'Debtor', 'Scholarship holder', 'Tuition fees up to date']
    from sklearn.metrics import recall_score, f1_score
    import pandas as pd

    subgroup_rows = []
    for col in subgroup_cols:
        if col in X_val.columns:
            for val in sorted(X_val[col].unique()):
                mask = (X_val[col] == val)
                if mask.sum() > 0:
                    y_sub_true = y_val[mask]
                    y_sub_pred = y_val_pred[mask]
                    rec_drop = recall_score(y_sub_true, y_sub_pred, labels=['Dropout'], average=None, zero_division=0)[0]
                    mf1 = f1_score(y_sub_true, y_sub_pred, average='macro', zero_division=0)
                    subgroup_rows.append({
                        'Feature': col,
                        'Group': val,
                        'Count': mask.sum(),
                        'Dropout Recall': round(rec_drop, 4),
                        'Macro-F1': round(mf1, 4)
                    })
    if subgroup_rows:
        df_sub = pd.DataFrame(subgroup_rows)
        print(df_sub.to_string(index=False))

if __name__ == "__main__":
    evaluate_model()
