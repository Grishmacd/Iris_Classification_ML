# Iris Flower Classification Using Logistic Regression (Machine Learning)

This project builds a **multi-class classification** model to predict the **Iris flower species** (Setosa, Versicolor, Virginica) using four flower measurements. It follows the complete ML workflow:
**Problem Statement → Selection of Data → Collection of Data → EDA → Train/Test Split → Model Selection → Evaluation Metrics**

---

## Problem Statement
Given flower measurements, predict which **Iris species** the flower belongs to.

**Input Features:**
- Sepal length
- Sepal width
- Petal length
- Petal width

**Output:**
- Species label: `setosa`, `versicolor`, `virginica`

---

## Selection of Data
**Dataset Type Used:** Structured tabular dataset (classic classification dataset)

Why this dataset is suitable:
- Clean, well-labeled dataset (3 classes)
- Perfect for practicing the full ML pipeline
- Numeric features make preprocessing and modeling straightforward

---

## Collection of Data
The dataset is loaded directly from Scikit-learn using:
- `sklearn.datasets.load_iris`

This provides ready-to-use feature data and target labels without manual downloading.

---

## EDA (Exploratory Data Analysis)
EDA is done to understand patterns between features and species:
- A combined DataFrame is created using `pd.concat([X, y], axis=1)`
- `seaborn.pairplot` is used to visualize how features separate species
- A correlation heatmap (`sns.heatmap(X.corr())`) is used to see relationships between features

This step helps visually confirm separability and feature relationships.

---

## Dividing Training and Testing
The dataset is split using `train_test_split`:
- Train set: model learns patterns
- Test set: model is evaluated on unseen data

Used in code:
- `test_size=0.2` (80% train, 20% test)
- `random_state=42` (reproducible results)

---

## Data Preprocessing
Features are scaled using **StandardScaler**:
- `fit_transform` on training data
- `transform` on test data

Why scaling is used:
- Helps Logistic Regression learn faster and perform more reliably
- Keeps all features on a similar scale

---

## Model Selection
**Model used:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)

Why Logistic Regression:
- Strong baseline for multi-class classification
- Simple, fast, and beginner-friendly

---

## Evaluation Metrics (Used in this Project)
This project evaluates the model using:
- **Confusion Matrix:** shows correct vs incorrect predictions for each class
- **Classification Report:** precision, recall, and F1-score for each class
- **Accuracy Score:** overall percentage of correct predictions

Used in code:
- `confusion_matrix(y_test, y_pred)`
- `classification_report(y_test, y_pred)`
- `accuracy_score(y_test, y_pred)`

---

## Main Libraries Used (and why)

1. `numpy`  
   - Supports numerical operations and array handling.

2. `pandas`  
   - Converts data into DataFrames/Series for easy viewing and manipulation.

3. `seaborn`  
   - Used for EDA visualizations like `pairplot` and heatmap.

4. `matplotlib.pyplot`  
   - Used to display plots and control plot styling.

5. `sklearn.datasets.load_iris`  
   - Loads the Iris dataset directly.

6. `sklearn.model_selection.train_test_split`  
   - Splits data into training and testing sets.

7. `sklearn.preprocessing.StandardScaler`  
   - Scales numeric features before training.

8. `sklearn.linear_model.LogisticRegression`  
   - Trains the classification model.

9. `sklearn.metrics`  
   - Evaluates model using confusion matrix, classification report, and accuracy.

---

## Output

- Pairplot and correlation heatmap for EDA
- Printed model results:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
  
---

## Developer
Grishma C.D
