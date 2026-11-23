# fitur.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan audit dasar pada data mentah."""
    print(f"Jumlah baris awal: {len(df)}")
    print(f"Missing values per kolom:\n{df.isnull().sum()}")
    return df

class PenguinFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer kustom untuk Feature Engineering sesuai aturan pipeline."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. Membuat fitur baru: Rasio paruh (Bill Ratio = Panjang Paruh / Kedalaman Paruh)
        if 'bill_length_mm' in X_copy.columns and 'bill_depth_mm' in X_copy.columns:
            X_copy['bill_ratio'] = X_copy['bill_length_mm'] / X_copy['bill_depth_mm']

        # 2. Membuang kolom yang tidak relevan: year
        if 'year' in X_copy.columns:
            X_copy = X_copy.drop(columns=['year'])

        return X_copy
