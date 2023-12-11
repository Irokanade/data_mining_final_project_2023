# Python 3.11.5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_disease.csv")

print(df.num.value_counts())

# sns.countplot(x="num", data=df, palette="bwr")
# plt.show()

df2 = pd.read_csv("heart2.csv")

print(df2.target.value_counts())

sns.countplot(x="num", data=df, palette="bwr")
plt.show()

countNoDisease = len(df[df.num == 0])
countHaveDisease = len(df[df.num != 0])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.num))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.num))*100)))

sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))

print(df.groupby('num').mean())

pd.crosstab(df.age,df.num).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()