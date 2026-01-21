import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# Sample Dataset (you can replace with CSV later)
# -----------------------------
data = {
    "country": ["USA", "USA", "UK", "UK", "India", "India", "Canada", "Canada"],
    "visa_type": ["Tourist", "Student", "Tourist", "Student", "Tourist", "Student", "Tourist", "Student"],
    "documents": [1, 1, 0, 1, 0, 1, 1, 0],
    "status": ["Approved", "Approved", "Rejected", "Approved", "Rejected", "Approved", "Approved", "Rejected"]
}

df = pd.DataFrame(data)

# Encode categorical data
le_country = LabelEncoder()
le_visa = LabelEncoder()
le_status = LabelEncoder()

df["country"] = le_country.fit_transform(df["country"])
df["visa_type"] = le_visa.fit_transform(df["visa_type"])
df["status"] = le_status.fit_transform(df["status"])

X = df[["country", "visa_type", "documents"]]
y = df["status"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "visa_model.pkl")
joblib.dump(le_country, "country_encoder.pkl")
joblib.dump(le_visa, "visa_encoder.pkl")
joblib.dump(le_status, "status_encoder.pkl")

print("✅ Model and encoders saved successfully!")
