'''------PART 1-------'''
import pandas as pd

#Read csv file in to a pandas dataframe
df = pd.read_csv("Divorces-PreFill.csv")

'''Back fill method'''
#Create dataframe for only Colorado's values
co_df = df.loc[df['State'] == 'Colorado']

#Use back fill method to populate the one missing value
co_df.fillna(method='bfill', inplace=True)

#Fill missing values in original dataframe with the newly filled colorado dratafram
df = df.fillna(co_df)


'''Forward fill method'''
#Create dataframe for the three states
ffill_df = df.loc[df['State'].isin(['Georgia', 'Louisiana', 'Hawaii'])]

#Use forward fill method to populate the three states
ffill_df.fillna(method='ffill', inplace=True)

#Fill original dataframe with the newly filled dataframe
df = df.fillna(ffill_df)

#Fill the rest of the missing values with each year's average
final_df = df.fillna(df.groupby(['Year']).transform('mean'))

#Write final dataframe to a csv file
final_df.to_csv('Divorces-PostFill.csv', index=False)


'''------PART 2------'''
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pandas import plotting as plt

import numpy as np

from sklearn import preprocessing as pp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics


#Read csv file in to a pandas dataframe
df = pd.read_csv("Crimes-PreFill.csv")


'''-----Data Frame prep for analysis-----'''

#Create a scale object
scale = pp.MinMaxScaler()
#Select columns to scale
columns = ['year', 'num_crimes']
#Fit the scale object to the columns & set values
df[columns] = scale.fit_transform(df[columns])
print('\nScaled:\n', df.head())

#Use the inverse transformation to get and set original values
df[columns] = scale.inverse_transform(df[columns])
print('\nReturned to Original:\n', df.head())

#Create dummy variable dataframe
df = pd.get_dummies(df, columns = ["Region", "Division", "State"], prefix_sep = '_')
print('\nDummy Variable Encoding:\n', df.head())


'''----Linear Regression accuracy evaluation-----'''

#Set seed so work can be reproduced
np.random.seed(100)
#Drop null dependent variables
no_null = df.dropna(how='any')
#Create dependent y for number of crimes
y = no_null['num_crimes']
#Create x. Drops last dummy variable
x = no_null.drop(['num_crimes', 'State_Wyoming', 'Region_West', 'Division_Mountain'], axis = 1)

#Split data in to train(80%) and test(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Build linear regression model on training data
lm_test = LinearRegression().fit(x_train, y_train)

#R2 evaluation of testing data
r2 = lm_test.score(x_test, y_test)
print('\nHold Out R2:', round(r2, 4))

#Make predictions with test teste
pred = lm_test.predict(x_test)
#Calculate RMSE of predictions and actual values (y) using the testing set
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print('Hold Out RMSE:', round(rmse, 4))


'''------ K-Fold Cross Validation -----'''

#Create empty linear regression model
lm_k = LinearRegression()

#Score cross validate R2 for each model generated with k-fold. k = 10
scores_r2 = list(np.round(cross_val_score(lm_k, x, y, cv=10), 4))
#Score cross validate mean squared error for each model generated with k-fold. k = 10
scores_mse = cross_val_score(lm_k, x, y, cv=10, scoring="neg_mean_squared_error")
#Score cross validate RMSE by taking square root of mse
scores_rmse = list(np.round(np.sqrt(np.abs(scores_mse)), 4))
print('\nCross-validated R2 By Fold:\n', scores_r2, "\nCross-Validated MSE By Fold:\n", scores_rmse)

#Get overal R2 and MSE for this model
#Generate prediction list
predictions = cross_val_predict(lm_k, x, y, cv=10)
#Compare the predicted vs actual values
r2 = metrics.r2_score(y, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y, predictions))
print('Cross-Validated R2 For All Folds:', round(r2, 4), '\nCross-Validation RMSE For All Folds:', round(rmse,4) )



'''---------Build Final Model and Prep Data for Output------'''

#Final model built with whole data set
lm_final = LinearRegression().fit(x, y)
#Get index number of null values
crime_null_list = df.index[df['num_crimes'].isnull()].tolist()
#Get x values for use 
pred_vals = df.drop(['num_crimes', 'State_Wyoming', 'Region_West', 'Division_Mountain'], axis=1)

#For every null value in crimes, fill with predicted value
#x = index of null value
for x in crime_null_list:               
    df.ix[x, 'num_crimes'] = lm_final.predict([pred_vals.iloc[x].tolist()])


'''--------Reshape dummy variables-------'''
#Create new dataframe of only columns
state = df.drop([col for col in df.columns if 'State' in col], axis=1)
region = df.drop([col for col in df.columns if 'Region' in col], axis=1)
division = df.drop([col for col in df.columns if 'Division' in col], axis=1)

#Drop all dummy variables
#df.drop(df.columns[2:], axis=1, inplace=True)

#Create new column name of the highest value for each dummy row - toke maybe as lambda

#Extract dataframe with only dummy column s
state = df[[col for col in df.columns if 'State' in col]]
region = df[[col for col in df.columns if 'Region' in col]]
division = df[[col for col in df.columns if 'Division' in col]]

#Drop Dummy variables
df.drop(df.columns[2:], axis=1, inplace=True)

#Extract name from column list and make column from it
df['State'] = state.idxmax(axis=1).str.extract('\_(.*)')
df['Region'] = region.idxmax(axis=1).str.extract('\_(.*)')
df['Division'] = division.idxmax(axis=1).str.extract('\_(.*)')

#Organize columns
df = df[["Region", "Division", "State", "year", "num_crimes"]]

print('\nFinal Dataset:\n',df.head())

#Create csv of filled 
df.to_csv('Crimes-PostFill.csv', index=False)























