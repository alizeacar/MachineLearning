import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
from sklearn.linear_model import LinearRegression
df = pd.read_csv('C:/Users/alize/Desktop/machine learning bootcamp/Salary_Data.csv')
print(df)
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
reg = LinearRegression()
reg.fit(X_Train, Y_Train)
Y_Pred = reg.predict(X_Test)
plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, reg.predict(X_Train), color = 'blue')
plt.title('Salary vs Experience  (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Salary vs Experience  (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
