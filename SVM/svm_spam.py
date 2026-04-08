from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

emails = []
labels = []

# Read email data from text file
with open("/home/prasanth/AIML/spam_emails.txt", "r") as file:
    for line in file:
        line = line.strip()

        # Skip empty lines
        if line == "":
            continue

        # Only process lines with comma
        if "," in line:
            text, label = line.split(",")

            emails.append(text)
            labels.append(int(label))

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# Kernels to test
kernels = ["linear", "poly", "rbf"]

for kernel in kernels:
    print("\n---------------------------")
    print("Kernel:", kernel)

    start_time = time.time()

    # Create SVM model
    model = SVC(kernel=kernel)

    # Train model
    model.fit(X_train, y_train)

    end_time = time.time()

    # Predict test data
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Training Time:", round(end_time - start_time, 5), "seconds")

import matplotlib.pyplot as plt

kernel_names = []
accuracy_values = []
training_times = []

for kernel in kernels:
    print("\n---------------------------")
    print("Kernel:", kernel)

    start_time = time.time()

    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    end_time = time.time()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    training_time = end_time - start_time

    kernel_names.append(kernel)
    accuracy_values.append(accuracy * 100)
    training_times.append(training_time)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Training Time:", round(training_time, 5), "seconds")

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.bar(kernel_names, accuracy_values)
plt.title("SVM Kernel Accuracy Comparison")
plt.xlabel("Kernel")
plt.ylabel("Accuracy (%)")
plt.show()

# Training Time Graph
plt.figure(figsize=(8, 5))
plt.bar(kernel_names, training_times)
plt.title("SVM Kernel Training Time Comparison")
plt.xlabel("Kernel")
plt.ylabel("Time (seconds)")
plt.show()
