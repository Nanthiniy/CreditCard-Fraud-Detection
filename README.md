Credit Card Fraud Detection Web Application

This project demonstrates the development of a Credit Card Fraud Detection Web Application using Machine Learning techniques. The application predicts the likelihood of a credit card transaction being fraudulent based on various features.

📘 Overview

Credit card fraud is a significant issue in the financial sector, leading to substantial losses annually. This application utilizes machine learning models to identify potentially fraudulent transactions, enhancing security measures for financial institutions and cardholders.

🚀 Features

User-Friendly Interface: Built with Streamlit for easy interaction.

Real-Time Prediction: Input transaction details and receive immediate fraud probability.

Model Integration: Utilizes a trained machine learning model for predictions.

Data Visualization: Displays relevant metrics and insights.

🧠 Tech Stack

Python 3.x

Streamlit: For building the web interface.

Scikit-learn: For machine learning model development.

Pandas & NumPy: For data manipulation and analysis.

Matplotlib & Seaborn: For data visualization.

📁 Project Structure

📦 credit-card-fraud-detection
 ┣ 📜 credit_card_fraud_prediction_app.py
 ┣ 📜 model.pkl
 ┣ 📜 requirements.txt
 ┣ 📜 README.md

    credit_card_fraud_prediction_app.py: Main application file.

    model.pkl: Serialized machine learning model.

    requirements.txt: Python dependencies.

🧩 How to Run

    Clone the repository

git clone https://github.com/Nanthiniy/CreditCard-Fraud-Detection.git

Navigate into the project directory

cd CreditCard-Fraud-Detection

Install dependencies

pip install -r requirements.txt

Run the application

    streamlit run credit_card_fraud_prediction_app.py

    Access the application

    Open your browser and go to http://localhost:8501 to use the application.

📊 Model Details

    Dataset: Utilizes the Credit Card Fraud Detection dataset from Kaggle

    .

    Preprocessing: Data normalization and handling of class imbalance.

    Model: Trained using Random Forest Classifier.

    Performance Metrics: Achieved high accuracy and recall, minimizing false negatives.

📈 Results

    Accuracy: 99.9%

    Precision: 98.5%

    Recall: 97.8%

    F1-Score: 98.1%

Note: These metrics are based on the evaluation of the model on the test dataset.
🧾 Future Enhancements

    Model Improvement: Experiment with other algorithms like XGBoost or Neural Networks.

    Real-Time Data Integration: Connect the application to live transaction data for real-time fraud detection.

    Deployment: Host the application on cloud platforms like Heroku or AWS for public access.

