import pytest
from app import app

def test_index_get():
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        assert b'Visa Type' in response.data

def test_index_post_valid():
    with app.test_client() as client:
        response = client.post('/', data={
            'visa_type': 'Tourist',
            'country': 'USA',
            'urgency': 'Low'
        })
        assert response.status_code == 200
        assert b'Estimated Processing Time' in response.data