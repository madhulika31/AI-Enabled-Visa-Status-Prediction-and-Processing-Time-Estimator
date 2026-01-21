from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
from models.preprocess import preprocess_input

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        visa_type = request.form.get('visa_type')
        country = request.form.get('country')
        urgency = request.form.get('urgency')
        
        # Server-side validation
        if not visa_type or not country or not urgency:
            flash('All fields are required.', 'error')
            return redirect(url_for('index'))
        
        # Prepare input
        input_data = pd.DataFrame({
            'visa_type': [visa_type],
            'country': [country],
            'urgency': [urgency]
        })
        
        # Preprocess
        processed_data = preprocess_input(input_data)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        
        # Redirect to results
        return render_template('result.html', prediction=round(prediction, 2))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
