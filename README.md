# ❤️ Heart Disease Prediction

This project uses **Machine Learning** models to predict the presence of **heart disease** based on clinical and diagnostic data. It demonstrates exploratory data analysis, model training, evaluation, and tuning — a great end-to-end data science pipeline! 🧠📊

---

## 📁 Dataset

The dataset includes medical attributes such as:

- 👤 Age, Gender  
- ❤️ Chest pain type  
- 🩺 Resting blood pressure  
- 🧪 Cholesterol level  
- 🩸 Blood sugar, ECG results, Heart rate  
- 🏃‍♂️ Exercise-induced angina  
- 📈 ST depression, Slope of ST, Number of major vessels, and more

✅ **No missing values were found.**
---

## 🤖 Models Used

- 🔍 Logistic Regression  
- 🌲 Random Forest  
- 🧱 Support Vector Machine (SVM)

---

## ⚙️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/RashikaJ/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
python heart_disease_prediction.py
```

## 📊 Results

### ✅ Model Performance

| Model                 | Accuracy  | Precision | Recall | F1 Score |
|-----------------------|-----------|-----------|--------|----------|
| Logistic Regression   | **0.885** | 0.879     | 0.906  | 0.892    |
| Random Forest         | 0.836     | 0.844     | 0.844  | 0.844    |
| SVM                   | 0.705     | 0.667     | 0.875  | 0.757    |

📌 Logistic Regression performed the best with the highest accuracy and balanced precision/recall.

## 🏆 Model Tuning (Random Forest)

Using GridSearchCV, the best parameters found were:

```json
{
  "max_depth": 5,
  "min_samples_leaf": 2,
  "min_samples_split": 5,
  "n_estimators": 50
}
```
Resulting in an optimized accuracy of 81.8%.
---

### 📉 Confusion Matrix - Logistic Regression

![🧩 Confusion Matrix Logistic Regression](images/confusion_matrix_logreg.png)

---

### 🔥 Correlation Heatmap

![🔗 Correlation Heatmap](images/correlation_heatmap.png)

---

## 🧠 Project Features

- 📊 Data exploration and visualization using Seaborn & Matplotlib  
- 🤖 Model training: Logistic Regression, Random Forest, SVM  
- 🛠️ Hyperparameter tuning using `GridSearchCV` & `RandomizedSearchCV`  
- 🎯 Evaluation using accuracy, precision, recall, and F1 score  
- 🧱 Confusion matrix visualization for each model

---

## 🚀 Future Improvements

- 🌐 Deploy as a Streamlit app  
- 🧠 Add deep learning models  
- 🗃️ Expand the dataset and include more features

---

## 🙋‍♀️ Author

**Rashika Jain**

🔗 [GitHub](https://github.com/RashikaJ)  
🔗 [LinkedIn](www.linkedin.com/in/rashika-j)

---

⭐ *If you like this project, give it a star!*
