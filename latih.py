# latih.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from fitur import audit_data, PenguinFeatureEngineer

def main():
    # 1. Load Dataset
    data_path = 'data/kasus03_penguins.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan.")

    df = pd.read_csv(data_path)
    df = audit_data(df)

    X = df.drop(columns=['species'])
    y = df['species']

    # 2. Pembagian Data Train dan Test SEBELUM preprocessing/imputasi (Aturan Wajib)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Definisi Fitur Numerik & Kategorikal setelah Feature Engineering
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_ratio']
    categorical_features = ['island', 'sex']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Definisi Dua Algoritma untuk Dibandingkan
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    best_model_name = None
    best_score = 0.0
    best_pipeline = None

    print("\n--- Perbandingan Model dengan Cross-Validation (Data Train) ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('feature_engineer', PenguinFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Cross-validation 5-fold
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv = cv_scores.mean()
        print(f"{name} - Mean CV Accuracy: {mean_cv:.4f} (+/- {cv_scores.std():.4f})")

        if mean_cv > best_score:
            best_score = mean_cv
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nModel Terbaik Berdasarkan CV: {best_model_name} dengan Akurasi {best_score:.4f}")

    # 5. Training final pada seluruh data train dan evaluasi SATU KALI pada data test
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nEvaluasi pada Data Test:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Simpan Model ke folder models/ menggunakan joblib (Tanpa path absolut)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'penguin_pipeline.joblib')

    # Simpan pipeline beserta metadata kelas target
    artifact = {
        'pipeline': best_pipeline,
        'classes': list(best_pipeline.classes_),
        'metrics': {
            'test_accuracy': test_accuracy,
            'best_model': best_model_name
        }
    }
    joblib.dump(artifact, model_path)
    print(f"\nPipeline berhasil disimpan di {model_path}")

if __name__ == '__main__':
    main()
