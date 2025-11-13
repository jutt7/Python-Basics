#imports
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('./ML/titanic.csv')



# data cleaning
def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], inplace=True)

    fill_missing_ages(df)

      # Fare: global median
    if df['Fare'].isna().any():
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
    #covert gender
    df["Sex"] = df["Sex"].map({'male':1, 'female':0})

    #feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0 , 1, 0)

    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)

    return df

def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)

data = preprocess_data(data)
# data.info()
# print(data.isnull().sum())

# Create features / Target Variables (Make Flashcards)

x = data.drop(columns=["Survived"])
feature_columns = x.columns
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# ML Preprocesssing

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparamenter Tuning - KNN

def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


best_model = tune_model(X_train, y_train)


# Prediction and evaluate

def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix')
print(matrix)


# plot
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived", "Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()

def predict_sample_passenger(sample_path, scaler, model, feature_columns):
    with open(sample_path, "r") as file:
        passenger_data = json.load(file)

    sample_df = pd.DataFrame([passenger_data])
    sample_df["Survived"] = np.nan

    titanic_raw = pd.read_csv("./ML/titanic.csv")
    inference_df = pd.concat([titanic_raw, sample_df], ignore_index=True, sort=False)
    inference_df = preprocess_data(inference_df)

    sample_features = inference_df[inference_df["Survived"].isna()].drop(columns=["Survived"])
    sample_features = sample_features[feature_columns]
    scaled_features = scaler.transform(sample_features)
    prediction = model.predict(scaled_features)[0]
    label = "Survived" if prediction == 1 else "Did not survive"
    print(f"Sample passenger prediction ({sample_path}): {label}")

# plot_model(matrix)
predict_sample_passenger("./ML/titanic_sample.json", scaler, best_model, feature_columns)
