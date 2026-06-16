Heart Disease Prediction Using Active Learning and Ensemble Learning

Overview

This project presents a Heart Disease Prediction System that combines Active Learning and Ensemble Machine Learning techniques to improve prediction accuracy while reducing labeling effort. The system uses multiple UCI Heart Disease datasets and employs a committee of machine learning models including XGBoost, Random Forest, and Logistic Regression.

A Flask-based web application allows users to enter clinical parameters and receive heart disease risk predictions in real time.

Features

* Active Learning using Query by Committee (QBC)
* Diversity Sampling using K-Means Clustering
* Ensemble Learning with:

  * XGBoost
  * Random Forest
  * Logistic Regression
* Majority Voting Prediction
* Flask Web Interface
* Real-Time Heart Disease Risk Assessment



Dataset

The project uses the UCI Heart Disease Dataset:

1. Cleveland Dataset
2. Hungarian Dataset
3. Switzerland Dataset
4. VA Long Beach Dataset

The datasets are merged into a single dataset for training and evaluation.

 Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib

System Workflow

1. Load and merge UCI datasets
2. Preprocess data and handle missing values
3. Split data into training and testing sets
4. Apply Active Learning (QBC + KMeans)
5. Train committee models
6. Generate ensemble predictions using majority voting
7. Display results through Flask web application


Performance Metrics

The system is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC


Project Structure

Heart-Disease-Prediction/

├── app.py

├── model.py

├── requirements.txt

├── README.md

├── templates/

│ └── index.html

├── static/

│ └── style.css

├── dataset/

│ └── merged_heart_disease.csv

└── screenshots/



Installation

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Open your browser and visit:

http://127.0.0.1:5000


Results

The trained ensemble model predicts whether a patient is at risk of heart disease based on clinical attributes such as age, blood pressure, cholesterol level, chest pain type, ECG results, and other medical indicators.


Future Enhancements

* Mobile Application Integration
* Cloud Deployment
* Explainable AI (XAI)
* Multi-Disease Prediction
* Electronic Health Record Integration


Author

Final Year Major Project

Heart Disease Prediction Using Active Learning and Ensemble Machine Learning Models
