import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("large_product_dataset.csv")

# Features (X) and target (y)
X = df["Product Name"]
y = df["Shelf Life (Days)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "shelf_life_model.pkl")
print("✅ Model trained & saved as shelf_life_model.pkl")
