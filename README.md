Titanic Survival Predictor: A Beginner-Friendly Guide

📌 Overview

This project aims to predict the survival of passengers on the Titanic using machine learning. By analyzing key factors such as age, gender, passenger class, and more, we build a predictive model using the famous Titanic dataset.

📂 Project Structure

titanic_survival_prediction/
│── data/
│   ├── train.csv  # Training dataset
│   ├── test.csv   # Testing dataset
│── notebooks/
│   ├── exploratory_data_analysis.ipynb  # EDA and visualization
│   ├── model_training.ipynb  # Model training and evaluation
│── src/
│   ├── data_preprocessing.py  # Data processing functions
│   ├── model.py  # Model training and evaluation
│── README.md  # Project documentation
│── requirements.txt  # Dependencies

🔍 1. Data Exploration & Visualization

Understanding the Dataset

We first load the training and test datasets and explore key features:

Passenger Information: Age, Sex, Pclass, Ticket, Fare, Embarked

Survival Analysis: Identifying factors that influenced survival

Handling Missing Values

Age: Filled with the median value.

Embarked: Filled with the most frequent value (mode).

Fare: Filled with the median for the test set.

Data Visualizations

Survival Distribution: Visualized survival rates using a count plot.

Class & Gender Impact: Used bar plots to analyze survival by class and gender.

Age Distribution: Used histograms to understand age-related survival trends.

🔧 2. Feature Engineering

Data Cleaning

Handled missing values using median imputation.

Encoded categorical features:

Sex: Male = 0, Female = 1

Embarked: C = 0, Q = 1, S = 2

Feature Selection

Selected the most relevant features:

Pclass, Sex, Age, Embarked

🤖 3. Model Building & Evaluation

Model: Random Forest Classifier

Used a Random Forest Classifier to predict survival.

Data Splitting: Training (80%) & Validation (20%) split.

Pipeline: Used SimpleImputer for handling missing values.

Model Performance

Accuracy: 77%

Classification Report:

Precision, Recall, and F1-score indicate balanced model performance.

🔑 Key Insights

Women & Children had higher survival rates (as expected historically).

Higher-class passengers had better survival chances.

Feature Importance: Pclass, Sex, and Age were the most significant predictors.

🚀 Installation & Usage

Clone the repository:

git clone https://github.com/your-repo/titanic_survival.git
cd titanic_survival

Install dependencies:

pip install -r requirements.txt

Run the Jupyter notebooks for data analysis and model training.

📜 License

This project is open-source and available for educational purposes.

🔹 Happy Learning & Coding! 🚀
