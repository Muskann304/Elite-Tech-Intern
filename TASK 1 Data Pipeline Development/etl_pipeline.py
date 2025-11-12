import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
import os

def extract_data():
    print("Extracting data...")
    data = fetch_openml(name='titanic', version=1, as_frame=True)
    df = data.frame
    return df

def transform_data(df):
    print("Transforming data...")
    X = df.drop(columns=['survived'])
    y = df['survived']
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor

def load_data(X_transformed, y, output_path='processed_data.csv'):
    print("Loading data...")
    X_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed)
    X_df['target'] = y.reset_index(drop=True)
    X_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def run_etl():
    df = extract_data()
    X_transformed, y, _ = transform_data(df)
    load_data(X_transformed, y)

if __name__ == '__main__':
    run_etl()
