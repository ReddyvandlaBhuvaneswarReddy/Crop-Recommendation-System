import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import shap


data = pd.read_csv("dataset.csv")

X = data.drop("Crop", axis=1)
y = data["Crop"]
feature_names = X.columns

le = LabelEncoder()
y = le.fit_transform(y)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=7,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,
    random_state=7,
    stratify=y_temp
)

print("Training size:", len(X_train))
print("Validation size:", len(X_val))
print("Testing size:", len(X_test))


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

models = {
    "LR": LogisticRegression(C=1, max_iter=3000),
    "DT": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=4,
        random_state=7
    ),
    "KNN": KNeighborsClassifier(n_neighbors=10),
    "SVM": SVC(C=1, gamma='scale'),
    "RF": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=7
    ),
    "GB": GradientBoostingClassifier(
        n_estimators=60,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=7
    )
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

results = []
trained_models = {}

for name, model in models.items():
    print("Training:", name)

    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    kfold_acc = cv_scores.mean() * 100

    model.fit(X_train, y_train)
    trained_models[name] = model

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    results.append([
        name,
        round(kfold_acc, 2),
        round(train_acc, 2),
        round(val_acc, 2),
        round(test_acc, 2)
    ])

print("\nResults:")
print(pd.DataFrame(results, columns=["Model", "K-Fold Acc", "Train Acc", "Val Acc", "Test Acc"]))
