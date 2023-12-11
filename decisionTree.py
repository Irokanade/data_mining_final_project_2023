import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

accuracies = {}

df = pd.read_csv("heart2.csv")

y = df.target.values
x = df.drop(['target'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

acc = dtc.score(x_test, y_test)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))


# using cross validation
print('using cross validation')
accuracy = cross_val_score(dtc, x, y, cv=10, scoring="accuracy")
print(accuracy)
print(accuracy.mean()*100,'%')