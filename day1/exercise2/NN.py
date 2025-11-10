import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Data Loading and Processing
data = pd.read_excel('data.xlsx')
feature_names = list(data)
feature_names.remove('ID')
feature_names.remove('FeS-cluster')
feature_names.remove('Reactant')
feature_names.remove('deltaH')
feature_names.remove('forwardEa')
feature_names.remove('postEa')
feature_names.remove('Behavior')
flag_name = 'Behavior'
class_names = ['0', '1', '2']  
X = data[feature_names]
Y = data[flag_name].map({0:'0', 1:'1', 2:'2'})  # mapping labels

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y)

# Model training
model = MLPClassifier(alpha=0.001, activation='relu', solver='adam', random_state=55)
model.fit(X_train, y_train)

# Evaluation
y_test_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_test_pred):.2f}')

# Visualization with confusion_matrix 
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_test_pred, labels=class_names)

# Heatmap
ax = sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names, 
    yticklabels=class_names,
    annot_kws={'fontsize': 24}
)

ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted', fontsize=24)
plt.ylabel('True', fontsize=24)
plt.title('NeuralNetwork Confusion Matrix', fontsize=24)
plt.show()

# Evaluation and Error Analysis
y_test_pred = model.predict(X_test)


misclassified_idx = np.where(y_test != y_test_pred)[0]


if len(misclassified_idx) > 0:
    error_samples = data.iloc[y_test.iloc[misclassified_idx].index]
    error_samples = error_samples.copy()  
    error_samples['True Label'] = y_test.iloc[misclassified_idx].values
    error_samples['Predicted Label'] = y_test_pred[misclassified_idx]
    error_report = error_samples[['True Label', 'Predicted Label']]
    print("\nMisclassified Samples:")
    print(error_report)
    error_report.to_csv('misclassified_samples_NN.csv', index=False)
    print("\nSaved: misclassified_samples.csv")
else:
    print("\nAll samples classified correctly!")