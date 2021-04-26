import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn import (metrics, discriminant_analysis, linear_model)
from sklearn.neighbors import KNeighborsClassifier

wineRed = pd.read_csv(r'winequality-red.csv')
wineWhite = pd.read_csv(r'winequality-white.csv')
np.random.seed(7)
wineRed.dropna()
wineWhite.dropna()

#We can comment and uncomment different sections based on what we want to output if that works!
#At some point maybe we split up red and white wine analysis into two different files but with all the same code, it might be easier

#Printing correlation matrix for red wine
""" print(wineRed.head())
sns.pairplot(wineRed) 
plt.show()
corrMatrix = wineRed.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show() """

#Printing correlation matrix for white wine
""" corrMatrix = wineWhite.corr()
sns.heatmap(corrMatrix, annot=True) """

#Train test split with all predictors for quality
""" X = wineRed.drop('quality', axis=1)
y = wineRed['quality']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=7) """

#OLS summary to see which p values are significant
""" Xb = sm.add_constant(Xtrain)
Xtest = sm.add_constant(Xtest)
ols = sm.OLS(ytrain, Xb).fit()
print(ols.summary())
ypred = ols.predict(Xtest)
mse = metrics.mean_squared_error(ytest, ypred)
print("OLS MSE: ", mse) """
#From this model, it looks like volatile acidity, chlorides, free sulfur dioxide, total sulfur dioxide, pH, sulphates, and alcohol are significant

#Logistic Regression summary to see which p values are significant
#First need to convert wineRed quality variables to classifiers, so 1 if > 5 for good quality, and 0 if < 5
""" wineRedB = wineRed
wineRedB['quality'] = np.where(wineRedB['quality'] > 5, 1, 0) """

""" X = wineRedB.drop('quality', axis=1)
y = wineRedB['quality']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=7)
Xb = sm.add_constant(Xtrain)
Xtest = sm.add_constant(Xtest)
logit = sm.Logit(ytrain, Xb).fit()
print(logit.summary())
ypred = logit.predict(Xtest)
mse = metrics.mean_squared_error(ytest, ypred)
print("Logistic Regression MSE: ", mse) """
#From here we can see that volatile acidity, citric acid, chlorides, free sulfur dioxide, total sulfur dioxide, sulphates, and alcohol are all significant
#The MSE value is also much lower than it was for OLS

#QDA



