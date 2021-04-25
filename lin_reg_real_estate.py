import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Data/Real estate.csv')
print(data.head())

print(data.info())

print(data.describe())

new_data = data.drop(data.iloc[:,0:2], axis=1)

new_data.columns = ['house age','distance to MRT','stores','latitude','longitude','house price']

sns.scatterplot(data=new_data, x='stores', y='house price')

sns.scatterplot(data=new_data, x='distance to MRT', y='house price')

sns.scatterplot(data=new_data, x='house age', y='house price')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
real_estate = pd.DataFrame(scaler.fit_transform(new_data))

X = real_estate.iloc[:,:-1] # all rows except last
y = real_estate.iloc[:,-1] # last columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
lg = LinearRegression().fit(X_train, y_train ) # train and fit the model

print('Coefficient is ', lg.coef_)

y_pred = lg.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error, r2_score
print('Mean squared error ', mean_squared_error(y_test, y_pred))
print('r2 score ', r2_score(y_test,y_pred))

reg_data = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})

sns.regplot(x='Actual',y='Prediction',data=reg_data)

