import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from src.data import load_data, get_train_val_test_splits
from src.features import DropFutureFeatures, build_preprocessor

def train_model(model_type='logistic'):
    """
    Huấn luyện mô hình chuẩn cho Bài 3: Ma trận nhầm lẫn, các chỉ số và khả năng tổng quát hóa.
    Mô hình chính: Multinomial Logistic Regression (class_weight='balanced')
    Baseline: DummyClassifier (most_frequent)
    Mô hình mở rộng: RandomForestClassifier
    """
    print("Loading data...")
    df = load_data()
    
    print("Splitting data (60% Train, 20% Val, 20% Test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits(df)
    
    print("Building pipeline...")
    preprocessor = build_preprocessor(X_train)
    
    if model_type == 'logistic':
        print("Training Multinomial Logistic Regression (Chủ đề cốt lõi Bài 3)...")
        classifier = LogisticRegression(
            solver='lbfgs', 
            class_weight='balanced', 
            max_iter=3000, 
            random_state=42
        )
    elif model_type == 'dummy':
        print("Training Baseline DummyClassifier...")
        classifier = DummyClassifier(strategy='most_frequent')
    else:
        print("Training RandomForestClassifier...")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
    pipeline = Pipeline(steps=[
        ('dropper', DropFutureFeatures()),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    print("Fitting pipeline on Train set...")
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
