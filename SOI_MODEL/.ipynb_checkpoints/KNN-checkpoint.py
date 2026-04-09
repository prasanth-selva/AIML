import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("4. KNN.csv")

# -----------------------------
# 1. Identify Problem Type
# -----------------------------
print("Problem Type: Classification")
print("Target Variable: Outcome")

# -----------------------------
# 2. Basic Dataset Information
# -----------------------------
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x='Outcome', data=df)
plt.title("Outcome Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title("Glucose vs Outcome")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Outcome', y='BMI', data=df)
plt.title("BMI vs Outcome")
plt.show()

# -----------------------------
# 4. Prepare Data
# -----------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Build KNN Model
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("\n========== KNN RESULTS ==========")
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# -----------------------------
# 6. Build Neural Network Model
# -----------------------------
nn_model = Sequential()

nn_model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=10,
    verbose=1
)

y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype("int32")

nn_accuracy = accuracy_score(y_test, y_pred_nn)

print("\n========== NEURAL NETWORK RESULTS ==========")
print("Neural Network Accuracy:", nn_accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nn))

# -----------------------------
# 7. Compare Both Models
# -----------------------------
print("\n========== MODEL COMPARISON ==========")
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Neural Network Accuracy:", nn_accuracy)

if nn_accuracy > accuracy_score(y_test, y_pred_knn):
    print("Neural Network performed better.")
elif nn_accuracy < accuracy_score(y_test, y_pred_knn):
    print("KNN performed better.")
else:
    print("Both models performed equally.")
