import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

accuracies = {}

df = pd.read_csv("heart2.csv")

y = df.target.values
x = df.drop(['target'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)

acc = rf.score(x_test,y_test)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

# using cross validation
print('using cross validation')
accuracy = cross_val_score(rf, x, y, cv=10, scoring="accuracy")
print(accuracy)
print(accuracy.mean()*100,'%')