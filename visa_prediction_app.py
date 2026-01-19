from flask import Flask, render_template, request, flash
from model import predict_processing_time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        country = request.form.get('country')
        visa_type = request.form.get('visa_type')
        applicant_age = request.form.get('applicant_age')
        if not all([country, visa_type, applicant_age]):
            flash('All fields are required.')
        else:
            try:
                applicant_age = int(applicant_age)
                prediction = predict_processing_time(country, visa_type, applicant_age)
            except ValueError:
                flash('Invalid age.')
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
