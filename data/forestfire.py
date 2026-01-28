from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/forestfires.csv')

df['fire'] = (df['area'] > df['area'].median()).astype(int)
df = df.drop(columns=['area'])

le = LabelEncoder()

df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])

X = df.drop(columns=['fire'])
y = df['fire']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42, stratify=y)

model = RandomForestClassifier(n_estimators = 100, random_state= 42)

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
report = classification_report(y_test,pred)
f1 = f1_score(y_test, pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("F1 Score:", f1)
print(y.value_counts())

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot= True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.scatter(y_test, pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

