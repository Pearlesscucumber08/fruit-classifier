# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part 1: Customize the Dataset
# Features: [weight (grams), color (0=green, 1=yellow, 2=red, 3=purple), shape (0=round, 1=long, 2=oval)]
# Labels: 0=apple, 1=banana, 2=cherry, 3=grape, 4=watermelon, 5=strawberry

X = np.array([
    [150, 2, 0],   # apple
    [120, 1, 1],   # banana
    [10, 2, 0],    # cherry
    [130, 2, 0],   # apple
    [110, 1, 1],   # banana
    [5, 2, 0],     # cherry
    [5, 3, 0],     # orange
    [2000, 0, 0],  # grapes
    [15, 2, 2]     # rambutan
])

y = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5]) 

# Part 2: Train the Model
model = DecisionTreeClassifier()
model.fit(X, y)
print("Model training complete.")

# Part 3: Test Your Classifier
test_fruits = np.array([
    [8, 3, 0],     # Likely orange
    [1800, 0, 0],  # Likely a grapes
    [12, 2, 2]     # Likely a rambutan
])

predictions = model.predict(test_fruits)

# Map labels to fruit names
fruit_names = {
    0: "apple",
    1: "banana",
    2: "cherry",
    3: "orange",
    4: "grapes",
    5: "rambutan"
}

for i, pred in enumerate(predictions):
    print(f"Test fruit {i+1} is predicted to be: {fruit_names[pred]}")

# Part 4: Bonus - Evaluate Accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_bonus = DecisionTreeClassifier()
model_bonus.fit(X_train, y_train)
y_pred = model_bonus.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data: {accuracy:.2f}")
