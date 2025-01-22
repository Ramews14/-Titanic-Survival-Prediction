# -Titanic-Survival-Prediction
This project predicts passenger survival on the Titanic using the famous Titanic dataset from Kaggle's Titanic Competition. By leveraging data preprocessing techniques, feature engineering, and machine learning, we built a Feedforward Neural Network (FNN) to achieve an 80% accuracy in survival prediction.

# Titanic Survival Prediction Using Logistic Regression and Neural Networks

## Project Overview
This project focuses on predicting the survival of passengers on the Titanic using logistic regression and a feedforward neural network. The dataset is sourced from Kaggle's Titanic competition and contains information about the passengers, such as their age, gender, class, and other characteristics. The goal is to create predictive models that determine whether a passenger is likely to survive based on these features.

## Dataset
The dataset consists of the following key features:

- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger (years)
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Passenger fare (British pounds)
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- `Survived`: Class variable (0 = did not survive, 1 = survived)

The `Survived` column serves as the target variable for model training.

## Objective
The objective is to use machine learning to predict passenger survival based on demographic, socioeconomic, and ticket information.

## Project Workflow

1. **Data Loading**: 
   - The dataset is loaded into a pandas DataFrame.
   
2. **Data Preprocessing**:
   - Handled missing values for `Age`, `Fare`, and `Embarked`.
   - Created new features, including `FamilySize` and `IsAlone`.
   - One-hot encoded categorical variables such as `Sex` and `Embarked`.
   - Scaled numerical features using standard scaling.

3. **Model Building**:
   - **Logistic Regression**: Used as a baseline to identify important features.
   - **Feedforward Neural Network**: Built using TensorFlow/Keras to improve predictions.
   
4. **Model Evaluation**:
   - Accuracy score to evaluate the models.
   - Plots of loss and accuracy during training for the neural network.
   - Confusion Matrix to analyze correct and incorrect predictions.

5. **Prediction**: 
   - The models predict whether a passenger survived (`1`) or did not survive (`0`) based on the features.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git


#### **requirements.txt:**
```plaintext
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow

