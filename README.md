# AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator
Deployed Url:https://ai-visa-status-prediction.streamlit.app/
## Project Overview
This project aims to predict visa application processing time using historical data and machine learning techniques.  
It helps applicants estimate how long their visa application may take based on factors such as visa type, country, application date, and processing center.
## Milestone 1: Data Collection & Preprocessing

### Objective
Build a clean, structured dataset suitable for analysis and machine learning modeling.

### Tasks Performed
- Collected historical/sample visa application data (public or synthetic).
- Identified important features:
  - Application date  
  - Decision date  
  - Visa type  
  - Country  
  - Processing center  
- Handled missing values using appropriate techniques.
- Converted date fields into proper datetime formats.
- Encoded categorical variables (Label Encoding / One-Hot Encoding).
- Created the target variable:
  - **Processing Time (in days)** = Decision Date − Application Date.

### Output
- Cleaned and preprocessed dataset
- Target variable ready for modeling

---

## Milestone 2: Exploratory Data Analysis (EDA) & Feature Engineering

### Objective
Understand data patterns and engineer meaningful features to improve model performance.

### Tasks Performed
- Conducted Exploratory Data Analysis (EDA) using:
  - Histograms
  - Box plots
  - Bar charts
  - Correlation heatmaps
- Analyzed:
  - Distribution of visa processing times
  - Differences across visa types and regions
  - Seasonal trends and workload patterns
- Identified correlations between features and processing time.
- Engineered new features:
  - Seasonal index (month/quarter-based)
  - Country-wise average processing time
  - Center-specific workload indicators

### Output
- EDA visualizations and insights
- Enhanced dataset with engineered features

---

## Milestone 3: Predictive Modeling

### Objective
Develop and evaluate machine learning models to predict visa processing time.

### Models Implemented
- Linear Regression
- Random Forest Regression

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Tasks Performed
- Split data into training and testing sets.
- Trained baseline and advanced regression models.
- Compared models using evaluation metrics.
- Performed hyperparameter tuning to improve performance.
- Selected the best-performing model for deployment.

### Output
- Trained regression models
- Model performance comparison
- Final optimized prediction model


## Milestone 4: Web Application Development & Deployment

### Objective
To develop, integrate, and deploy a web-based application that utilizes the trained regression model to predict visa status and estimate visa processing time in real time.

---

### Tasks Performed
- Designed and developed a user-friendly web interface using **Flask / Streamlit**.
- Created structured input forms to capture visa application details.
- Integrated the trained **regression model** from Milestone 3 into the backend.
- Implemented prediction logic for:
  - Visa approval status
  - Estimated processing time (in days)
- Ensured seamless communication between frontend and backend components.
- Configured application dependencies and runtime environment.
- Deployed the application on a **cloud platform** (AWS / Azure / Heroku/Render/Netfly).
- Performed final functional testing using multiple sample test cases.
- Validated system performance, accuracy, and response time.

---

### Output
- Fully functional AI-enabled web application.
- Real-time prediction of visa status and processing time.
- Successfully deployed cloud-based solution accessible via web browser.
- Stable and scalable system ready for real-world usage.
---

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- APIs

---

## Project Status
✅ Milestone 1 - Completed 
✅ Milestone 2 - Completed 
✅ Milestone 3 - Completed   
✅ Milestone 4 - Completed

---
🚀 Future Enhancements:-
*Use advanced ML models to improve prediction accuracy.
*Integrate real-time visa application data.
*Create separate models for different countries and visa types.
*Add user login and application history tracking.
*Provide explanation for predictions using Explainable AI.
*Develop a mobile app version of the system.
*Add multilingual support and chatbot assistance.
✅ Conclusion:-
The AI-Enabled Visa Status Prediction and Processing Time Estimator helps users predict visa approval chances and expected processing time using machine learning. 
It reduces uncertainty and supports better planning for applicants. 
The project shows how AI can be applied in real-world immigration-related problems and provides a strong base for future improvements using real-time data and advanced models.

## Author
**Madhulika Thripuravaram**
