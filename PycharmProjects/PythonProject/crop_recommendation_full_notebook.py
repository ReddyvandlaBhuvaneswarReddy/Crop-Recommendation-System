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

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "KFold Accuracy",
        "Training Accuracy",
        "Validation Accuracy",
        "Testing Accuracy"
    ]
)

print("\nModel Accuracy Comparison\n")
print(results_df)

best_index = results_df["Testing Accuracy"].idxmax()
best_model_name = results_df.loc[best_index, "Model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)
print("Final Accuracy:", results_df.loc[best_index, "Testing Accuracy"], "%")

from sklearn.model_selection import GridSearchCV

dt_params = {
    "max_depth": [5, 8, 10, 12],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
    "criterion": ["gini", "entropy"]
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=7),
    param_grid=dt_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_

print("\nBest Params:", grid_dt.best_params_)

print("\nSHAP Explanation")

X_sample = X_test[:100]

explainer = shap.TreeExplainer(best_dt)
shap_values = explainer(X_sample)

vals = shap_values.values

if len(vals.shape) == 3:
    shap_vals = np.mean(np.abs(vals), axis=2)
else:
    shap_vals = np.abs(vals)
print("SHAP shape:", shap_vals.shape)
print("X_sample shape:", X_sample.shape)

shap.summary_plot(
    shap_vals,
    X_sample,
    feature_names=feature_names
)

importance = np.mean(shap_vals, axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print("\nFeature Importance\n")
print(importance_df)

metric_results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    metric_results.append([
        name,
        round(precision, 2),
        round(recall, 2),
        round(f1, 2)
    ])

metrics_df = pd.DataFrame(
    metric_results,
    columns=[
        "Model",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nPRECISION / RECALL / F1\n")
print(metrics_df)

from sklearn.metrics import classification_report

print("\nCLASSIFICATION REPORT \n")
y_pred_dt = best_dt.predict(X_test)
print(classification_report(y_test, y_pred_dt))

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

y_score = best_dt.predict_proba(X_test)

classes = le.classes_

plt.figure(figsize=(10, 8))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'Class {classes[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

plt.title("ROC Curve for Decision Tree Model (All Classes - Zoomed In)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.xlim([0.0, 0.2])
plt.ylim([0.8, 1.0])

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_best_model = best_dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best_model)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Best Model (Decision Tree)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=[str(i) for i in np.unique(y)],
    mode='classification'
)

exp = explainer.explain_instance(
    X_test[0],
    best_dt.predict_proba
)

print("\nLIME Explanation:")
exp.show_in_notebook(show_table=True)

# Additional notebook cells

data.head()

plt.figure(figsize=(8,5))
x = results_df["Model"]
y = results_df["Testing Accuracy"]

plt.bar(x, y)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

for i, val in enumerate(y):
    plt.text(i, val, f"{val:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
x = metrics_df["Model"]
y = metrics_df["Precision"]

plt.bar(x, y)
plt.title("Model Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision (%)")

for i, val in enumerate(y):
    plt.text(i, val, f"{val:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
x = metrics_df["Model"]
y = metrics_df["Recall"]

plt.bar(x, y)
plt.title("Model Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall (%)")

for i, val in enumerate(y):
    plt.text(i, val, f"{val:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
x = metrics_df["Model"]
y = metrics_df["F1 Score"]

plt.bar(x, y)
plt.title("Model F1 Score Comparison")
plt.xlabel("Models")
plt.ylabel("F1 Score (%)")

for i, val in enumerate(y):
    plt.text(i, val, f"{val:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nF1 Score Compilation:")
print(metrics_df)
