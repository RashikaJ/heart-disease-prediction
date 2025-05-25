import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

data = pd.read_csv('heart.csv')
print(data.head())
print(data.isnull().sum())
print(data.dtypes)
print(data.describe())

sns.countplot(x='target', data=data)
plt.title('Heart Disease Presence Distribution')
plt.show()

sns.boxplot(x='target', y='age', data=data)
plt.title('Age Distribution by Heart Disease Presence')
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

#crosstab = pd.crosstab(data['exang'], data['target'])
#print(crosstab)

x = data.drop("target", axis=1)
y = data["target"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

from sklearn.svm import SVC

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

from sklearn.metrics import classification_report, accuracy_score

print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))
print("SVM Report:\n", classification_report(y_test, y_pred_svm))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(x_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Logistic Regression": model,
    "Random Forest": rf,
    "SVM": svm
}

results = []

for name, model in models.items():
    y_pred = model.predict(x_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)

import seaborn as sns
import matplotlib.pyplot as plt

results_df.set_index("Model").plot(kind="bar", figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=20,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42)

random_search.fit(x_train, y_train)
print("Best parameters found:", random_search.best_params_)
print("Best accuracy:", random_search.best_score_)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, clf) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test, ax=ax, cmap='Blues', colorbar=False)
    ax.title.set_text(f'Confusion Matrix - {name}')
plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

param_dist_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm = SVC(probability=True, random_state=42)
random_search_svm = RandomizedSearchCV(estimator=svm,
                                       param_distributions=param_dist_svm,
                                       n_iter=15,
                                       scoring='accuracy',
                                       cv=5,
                                       n_jobs=-1,
                                       random_state=42)

random_search_svm.fit(x_train, y_train)
print("Best parameters for SVM:", random_search_svm.best_params_)
print("Best accuracy:", random_search_svm.best_score_)

