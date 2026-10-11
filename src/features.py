from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DropFutureFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop 1st semester and 2nd semester features
    because the prediction time is at enrollment.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        cols_to_drop = [col for col in X_copy.columns if '1st sem' in col or '2nd sem' in col]
        return X_copy.drop(columns=cols_to_drop)

def build_preprocessor(X_train):
    """
    Builds the ColumnTransformer for the pipeline.
    """
    # Temporarily drop future features to see what's left
    cols_to_drop = [col for col in X_train.columns if '1st sem' in col or '2nd sem' in col]
    X_reduced = X_train.drop(columns=cols_to_drop)
    
    numeric_features = [
        'Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    categorical_features = [col for col in X_reduced.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    return preprocessor
