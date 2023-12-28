import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

accuracies = {}

df = pd.read_csv("heart2.csv")

y = df.target.values
x = df.drop(['target'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

estimators_range = range(1, 100)

# Dictionary to store accuracies for different max_depth values
accuracies = {}

for estimator_n in estimators_range:
    print(f'estimator: {estimator_n}')
    rf = RandomForestClassifier(n_estimators = estimator_n, random_state = 1)
    accuracy = cross_val_score(rf, x, y, cv=10, scoring="accuracy")
    accuracies[estimator_n] = accuracy.mean()

best_n_estimator = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_n_estimator]

print(f"Best Max Depth: {best_n_estimator}")
print(f"Corresponding Accuracy: {best_accuracy * 100:.2f}%")

# Plot the results
plt.plot(estimators_range, [accuracies[d] for d in estimators_range], marker='o')
plt.title('Cross-Validation Accuracy vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.show()

# rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
# rf.fit(x_train, y_train)

# acc = rf.score(x_test,y_test)*100
# accuracies['Random Forest'] = acc
# print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

# # using cross validation
# print('using cross validation')
# accuracy = cross_val_score(rf, x, y, cv=10, scoring="accuracy")
# print(accuracy)
# print(accuracy.mean()*100,'%')