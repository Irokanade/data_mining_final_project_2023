import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("heart2.csv")

# Split the data into features (x) and target variable (y)
y = df.target.values
x = df.drop(['target'], axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define a range of maximum depth values
max_depth_values = range(1, 101)

# Dictionary to store accuracies for different max_depth values
accuracies = {}

# Iterate over different max_depth values
for max_depth in max_depth_values:
    # Create DecisionTreeClassifier with the current max_depth value
    dtc = DecisionTreeClassifier(max_depth=max_depth, splitter='random')
    
    # Perform cross-validation
    accuracy = cross_val_score(dtc, x, y, cv=10, scoring="accuracy")
    
    # Store the mean accuracy for the current max_depth
    accuracies[max_depth] = accuracy.mean()

# Find the max_depth value with the highest mean accuracy
best_max_depth = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_max_depth]

# Print the best max_depth and accuracy
print(f"Best Max Depth: {best_max_depth}")
print(f"Corresponding Accuracy: {best_accuracy * 100:.2f}%")

# Plot the results
plt.plot(max_depth_values, [accuracies[d] for d in max_depth_values], marker='o')
plt.title('Cross-Validation Accuracy vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.show()
