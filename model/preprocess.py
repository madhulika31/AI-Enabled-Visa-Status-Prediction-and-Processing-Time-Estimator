import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encoders (fit on training data)
visa_encoder = LabelEncoder()
country_encoder = LabelEncoder()
urgency_encoder = LabelEncoder()
scaler = StandardScaler()

def preprocess_data(data):
    # Encode categoricals
    data['visa_type_encoded'] = visa_encoder.fit_transform(data['visa_type'])
    data['country_encoded'] = country_encoder.fit_transform(data['country'])
    data['urgency_encoded'] = urgency_encoder.fit_transform(data['urgency'])
    
    # Features and target
    X = data[['visa_type_encoded', 'country_encoded', 'urgency_encoded']]
    y = data['processing_time']
    
    # Scale features
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def preprocess_input(input_data):
    # Encode using fitted encoders
    input_data['visa_type_encoded'] = visa_encoder.transform(input_data['visa_type'])
    input_data['country_encoded'] = country_encoder.transform(input_data['country'])
    input_data['urgency_encoded'] = urgency_encoder.transform(input_data['urgency'])
    
    X = input_data[['visa_type_encoded', 'country_encoded', 'urgency_encoded']]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    return X_scaled