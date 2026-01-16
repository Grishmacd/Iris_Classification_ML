# Iris_Classification_ML

This project builds a Machine Learning model to classify **Iris flowers** into one of three species:
1. Setosa
2. Versicolor
3. Virginica

It follows the complete ML workflow:

**Problem Statement → Data Selection → Data Collection → EDA → Train/Test Split → Model Selection → Evaluation Metrics**

---

## Problem Statement
Classify an Iris flower into its correct species based on these input features:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

**Output:** Predicted Iris class (3-category classification)

---

## Selection of Data
**Dataset chosen:** Iris Dataset  
**Why this dataset?**
- Clean and small dataset (good for learning ML pipeline)
- Clearly labeled 3-class classification problem
- Numerical features (easy to preprocess and train)

---

## Collection of Data
The dataset is loaded directly from **scikit-learn** built-in datasets:
- `sklearn.datasets.load_iris`

This avoids manual downloading and ensures the dataset is standardized.

---

## Main Libraries Used (and why)
1. `pandas`  
   - For tabular data handling (DataFrame), inspection, and basic analysis.

2. `numpy`  
   - For numerical operations, arrays, and shape handling.

3. `matplotlib.pyplot`  
   - For plotting graphs (visual understanding in EDA).

4. `seaborn`  
   - For better-looking statistical plots during EDA.

5. `sklearn.datasets.load_iris`  
   - To load the Iris dataset directly.

6. `sklearn.model_selection.train_test_split`  
   - To split dataset into training and testing parts.

7. `sklearn.preprocessing.StandardScaler`  
   - To scale features to a standard range (important for models like Logistic Regression).

8. `sklearn.linear_model.LogisticRegression`  
   - The ML model used for classification.

9. `sklearn.metrics (accuracy_score, confusion_matrix, classification_report)`  
   - To evaluate model performance.

---

## Overall Project Flow (Step-by-step)

### 1) Import Libraries
First we import the required libraries for:
- Data handling (pandas, numpy)
- Visualization (matplotlib, seaborn)
- ML pipeline (sklearn)

### 2) Load the Dataset
- Load Iris data using `load_iris()`
- Convert it into a DataFrame for easier viewing and EDA

### 3) EDA (Exploratory Data Analysis)
Goal: understand the dataset before training.
Typical EDA includes:
1. Checking dataset shape (rows, columns)
2. Viewing first few rows
3. Checking missing values
4. Checking class distribution (how many samples per class)
5. Visualizing relationships between features (optional graphs)

### 4) Define X and y
- `X` = input features (sepal/petal measurements)
- `y` = target labels (flower species)

### 5) Divide into Training and Testing
Using `train_test_split`:
- Training set: used to learn patterns
- Testing set: used to evaluate final performance

This prevents the model from “memorizing” the dataset and gives a real measure of performance.

### 6) Preprocessing (Scaling)
Using `StandardScaler`:
- Fit scaler on training data
- Transform both train and test data

Why scaling?
- Keeps features on similar scale
- Helps Logistic Regression perform better and converge faster

### 7) Model Selection
**Model used:** `LogisticRegression`

Why Logistic Regression?
- Works well for classification
- Fast and simple baseline model
- Good for multi-class classification (like Iris)

### 8) Train the Model
- Fit the model on training data:
  - `model.fit(X_train, y_train)`

### 9) Predictions
- Predict on test set:
  - `y_pred = model.predict(X_test)`

### 10) Evaluation Metrics (Evaluation Matrix)
To judge model performance, we use:

1. **Accuracy Score**
- Overall correctness of predictions:
  - `accuracy_score(y_test, y_pred)`

2. **Confusion Matrix**
- Shows correct vs incorrect predictions per class:
  - `confusion_matrix(y_test, y_pred)`

3. **Classification Report**
Includes:
- Precision
- Recall
- F1-score
- Support
  - `classification_report(y_test, y_pred)`

---

## Model Used (Important)
**Logistic Regression (from scikit-learn)**
- `from sklearn.linear_model import LogisticRegression`

This is the main ML model used in your notebook.

---
## Developer
Grishma C.D
