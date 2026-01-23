import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------ Load Dataset ------------------
# Your dataset file name should be visa_dataset.csv
df = pd.read_csv("visa_dataset.csv")

# ------------------ Encoding ------------------
country_map = {"India": 0, "USA": 1, "UK": 2, "Canada": 3, "Australia": 4}
visa_map = {"Student": 0, "Work": 1, "Tourist": 2, "Dependent": 3}
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
status_map = {"Rejected": 0, "Approved": 1}

df["Country"] = df["Country"].map(country_map)
df["VisaType"] = df["VisaType"].map(visa_map)
df["Education"] = df["Education"].map(edu_map)
df["VisaStatus"] = df["VisaStatus"].map(status_map)

# ------------------ Features & Target ------------------
X = df[["Age", "Country", "VisaType", "Experience", "Education"]]
y = df["VisaStatus"]

# ------------------ Train Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Train Model ------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------ Evaluation ------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# ------------------ Save Model ------------------
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
