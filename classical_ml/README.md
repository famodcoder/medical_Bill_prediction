# 🏥 Medical Insurance Bill Predictor

Welcome to the end-to-end Machine Learning deployment project! This repository demonstrates how to take a trained classical machine learning model and deploy it as a fully interactive web application.

## 🎯 Learning Objectives
In this project, we bridge the gap between model training and real-world application. You will see how to:
- Serialize a trained model using `joblib`.
- Handle categorical user inputs in a production environment (bypassing the `pd.get_dummies` trap).
- Build a user-friendly frontend using **Streamlit**.

## 📂 Repository Structure
- `notebooks/`: Contains `InsurancePrediction.ipynb`, where the data was cleaned, explored, and the model was trained.
- `models/`: Contains the serialized (pickled) model and the exact column structures required for making predictions.
- `app.py`: The Streamlit web application script that creates our user interface.
- `requirements.txt`: The specific Python libraries needed to run this project.

## 🚀 How to Run the App Locally

If you want to run this web app on your own computer, follow these steps:

**1. Clone the repository**
```bash
git clone <paste-your-github-repo-url-here>
cd classic-ml-deployment