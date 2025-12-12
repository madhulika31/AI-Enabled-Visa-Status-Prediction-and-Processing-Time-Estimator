import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('visa_processing_dataset_upgraded_50k.csv')

# Label encoding for categorical features
le_visa_type = LabelEncoder()
le_country = LabelEncoder()
le_processing_center = LabelEncoder()

df['visa_type_encoded'] = le_visa_type.fit_transform(df['visa_type'])
df['country_encoded'] = le_country.fit_transform(df['country'])
df['processing_center_encoded'] = le_processing_center.fit_transform(df['processing_center'])

# Drop original categorical columns and keep encoded ones
df_processed = df[['visa_type_encoded', 'country_encoded', 'processing_center_encoded', 'processing_time_days']]

# Save processed data
df_processed.to_csv('processed_visa.csv', index=False)
print("Data preprocessed and saved as processed_visa.csv")
