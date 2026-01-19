import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Dummy data generation
def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    countries = ['USA', 'Canada', 'UK', 'Australia', 'Germany']
    visa_types = ['Tourist', 'Business', 'Student', 'Work']
    data = {
        'country': np.random.choice(countries, n_samples),
        'visa_type': np.random.choice(visa_types, n_samples),
        'applicant_age': np.random.randint(18, 65, n_samples),
        'processing_days': np.random.randint(10, 180, n_samples)  # Target
    }
    return pd.DataFrame(data)

# Preprocessing
def preprocess_data(df):
    le_country = LabelEncoder()
    le_visa = LabelEncoder()
    df['country_encoded'] = le_country.fit_transform(df['country'])
    df['visa_type_encoded'] = le_visa.fit_transform(df['visa_type'])
    scaler = StandardScaler()
    df[['applicant_age', 'country_encoded', 'visa_type_encoded']] = scaler.fit_transform(df[['applicant_age', 'country_encoded', 'visa_type_encoded']])
    return df, le_country, le_visa, scaler

# Train model
def train_model():
    df = generate_dummy_data()
    df, le_country, le_visa, scaler = preprocess_data(df)
    X = df[['applicant_age', 'country_encoded', 'visa_type_encoded']]
    y = df['processing_days']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')
    # Save model and preprocessors
    joblib.dump(model, 'model.joblib')
    joblib.dump(le_country, 'le_country.joblib')
    joblib.dump(le_visa, 'le_visa.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    return model, le_country, le_visa, scaler

# Load model
def load_model():
    model = joblib.load('model.joblib')
    le_country = joblib.load('le_country.joblib')
    le_visa = joblib.load('le_visa.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, le_country, le_visa, scaler

# Predict
def predict_processing_time(country, visa_type, applicant_age):
    model, le_country, le_visa, scaler = load_model()
    country_encoded = le_country.transform([country])[0]
    visa_encoded = le_visa.transform([visa_type])[0]
    features = scaler.transform([[applicant_age, country_encoded, visa_encoded]])
    prediction = model.predict(features)[0]
    return max(0, int(prediction))  # Ensure non-negative

if __name__ == '__main__':
    train_model()
