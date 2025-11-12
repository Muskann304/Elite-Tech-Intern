import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df = pd.read_csv('data/train.csv')
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
y = df['Survived']

# Define preprocessing
num_features = ['Age', 'SibSp', 'Fare']
cat_features = ['Pclass', 'Sex']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Final pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, 'model.joblib')
print("Model saved as model.joblib")
