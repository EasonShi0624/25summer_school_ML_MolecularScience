# Exercise 2: Classification Algorithms

## 1. Environment Setup & Library Imports

1. **Create a Python 3.9 virtual environment**

   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

2. **Install required packages**

   ```bash
   pip install scikeras seaborn scikit-learn pandas matplotlib
   ```

3. **Import libraries in your script or notebook**

   ```python
   import numpy as np
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.svm import SVC
   from sklearn.neural_network import MLPClassifier
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
   ```

------

## 2. Data Loading & Preprocessing

1. **Load the dataset** (replace with your actual path)

   ```python
   df = pd.read_csv('path/to/your/dataset.csv')
   ```

2. **Define feature and target columns**

   ```python
   feature_names = ['feat1', 'feat2', 'feat3', …]   # e.g. column names
   target_name   = 'Behavior'
   ```

3. **Map numeric labels to human-readable classes**

   ```python
   label_map = {1: 'Class A', 2: 'Class B', 3: 'Class C'}
   df['label'] = df[target_name].map(label_map)
   ```

4. **Split into features (X) and labels (y)**

   ```python
   X = df[feature_names]
   y = df['label']
   ```

5. **Train-test split** (80:20 by default; adjust `test_size` as needed)

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

------

## 3. Model Construction & Training

### 3.1 Support Vector Machine (SVM)

```python
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)
```

### 3.2 Neural Network (Multi-Layer Perceptron)

```python
nn = MLPClassifier(
    hidden_layer_sizes=(100, 50), activation='relu',
    solver='adam', max_iter=300, random_state=42
)
nn.fit(X_train, y_train)
```

### 3.3 Random Forest (RF)

```python
rf = RandomForestClassifier(
    n_estimators=100, max_depth=None,
    min_samples_split=2, random_state=42
)
rf.fit(X_train, y_train)
```

> **Tip:** Use `GridSearchCV` or `RandomizedSearchCV` to tune key hyperparameters (e.g. `C`, `gamma` for SVM; layer sizes, learning rate for NN; `n_estimators`, `max_depth` for RF).

------

## 4. Model Evaluation

1. **Make predictions**

   ```python
   y_pred = rf.predict(X_test)
   ```

2. **Compute accuracy**

   ```python
   acc = accuracy_score(y_test, y_pred)
   print(f"Test Accuracy: {acc:.2%}")
   ```

3. **Confusion matrix & classification report**

   ```python
   cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
   print("Classification Report:\n", classification_report(y_test, y_pred))
   ```

4. **Visualize confusion matrix**

   ```python
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=rf.classes_,
               yticklabels=rf.classes_)
   plt.xlabel('Predicted Label')
   plt.ylabel('True Label')
   plt.title('Confusion Matrix')
   plt.show()
   ```

> **Additional metrics:** consider ROC curves, AUC scores (for binary tasks) or precision-recall curves for imbalanced data.

------

## 5. Next Steps & Best Practices

- **Cross-validation:** Evaluate model stability (e.g. 5- or 10-fold CV) rather than a single split.
- **Feature importance:** For tree-based models, inspect `.feature_importances_`.
- **Scaling & preprocessing:** Apply standardization (`StandardScaler`) or normalization as needed, especially for SVM and NN.
- **Ensemble methods:** Combine multiple classifiers (e.g. voting, stacking) to boost performance.
- **Documentation & reproducibility:** Log hyperparameters, random seeds, and environment details.

