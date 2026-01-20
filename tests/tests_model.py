import pytest
import pandas as pd
from models.preprocess import preprocess_input
import pickle

def test_preprocess_input():
    input_data = pd.DataFrame({
        'visa_type': ['Tourist'],
        'country': ['USA'],
        'urgency': ['Low']
    })
    processed = preprocess_input(input_data)
    assert processed.shape == (1, 3)

def test_model_prediction():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    import numpy as np
    pred = model.predict(np.array([[0, 0, 0]]))
    assert isinstance(pred[0], (int, float))