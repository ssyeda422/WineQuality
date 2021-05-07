import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import (metrics, discriminant_analysis, linear_model)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import graphviz

#UNCOMMENT CORRECT FILE PATH - there is something wrong in
wineRed = pd.read_csv(r'winequality-red.csv')
np.random.seed(7)
wineRed.dropna()

#We can comment and uncomment different sections based on what we want to output if that works!
#At some point maybe we split up red and white wine analysis into two different files but with all the same code, it might be easier

#Printing correlation matrix for red wine
"""print(wineRed.head())
sns.pairplot(wineRed)"""

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
#But the mean squared error is pretty high, so we can also try a different model besides regression 

#We can try using the predictor alcohol to predict quality with polynomial regression
#Uncomment this section to use the X predictor as just alcohol for poly reg
""" wineRedB = wineRed.sort_values(by=['alcohol'])
X = wineRedB['alcohol']
X = X.values.reshape(-1,1)
y = wineRedB['quality']
np.random.seed(312) """

#KFold cross validation to find optimal polynomial degree:
""" kf = KFold(n_splits=10, shuffle=True, random_state=312)
kf_scores = pd.Series()
for i in range(1, 10):
    poly_reg = Pipeline([("poly", PolynomialFeatures(degree = i)), ("reg", linear_model.LinearRegression())])
    cv_scores = cross_val_score(poly_reg, X, y, scoring="neg_mean_squared_error", cv = kf)
    kf_scores.loc[i] = abs(cv_scores.mean())
print("Min: ", kf_scores.idxmin())
plt.plot(np.arange(1,10), kf_scores)  """
#Optimal degree seems to be 3, min MSE

#Plotting the curve
""" poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
lin_reg = linear_model.LinearRegression(fit_intercept=False)
lin_reg.fit(X_poly, y)
y_pred = lin_reg.predict(X_poly)
plt.plot(X, y_pred, color="blue")
plt.title('Polynomial Regression: Alcohol vs. Quality')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
print("Polynomial Regression: Alcohol vs. Quality MSE: ", metrics.mean_squared_error(y, y_pred)) """
#Although the curve shows a relationship, the MSE is still pretty high, so we can try classification

#We can convert wineRed quality variables to classifiers, so 1 if >= 7 for good quality, and 0 if < 7
#Uncomment this section to use the classification dataset, and comment out the previous train test split which doesn't transform the quality column
wineRedB = wineRed
wineRedB['quality'] = np.where(wineRedB['quality'] >= 7, 1, 0)

X = wineRedB.drop('quality', axis=1)
y = wineRedB['quality']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=7)

#Logistic Regression summary - Classification, using statsmodels
""" Xb = sm.add_constant(Xtrain)
Xtest = sm.add_constant(Xtest)
logit = sm.Logit(ytrain, Xb).fit()
print(logit.summary())
ypred = logit.predict(Xtest)
mse = metrics.mean_squared_error(ytest, ypred)
print("Logistic Regression MSE: ", mse) """

#QDA - Classification
"""qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda.fit(Xtrain, ytrain)
ypred = qda.predict(Xtest)
score = qda.score(Xtest, ytest)
print("\nQDA Score: ", score)
mse = metrics.mean_squared_error(ytest, ypred)
print("QDA MSE: ", mse)"""

#LDA - Classification 
"""lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(Xtrain, ytrain)
ypred = lda.predict(Xtest)
scores = cross_val_score(lda, Xtrain, ytrain)
lda_score = scores.mean()
print("\nLDA Score: ", lda_score) 
lda_mse  = metrics.mean_squared_error(ytest, ypred)
print("LDA MSE: ", lda_mse)"""

#KNN - Classification 
# I am unfamiliar with python so I followed this: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# Wanted to reference it just in case 
# compate MSE with different K values 
"""error = []
for i  in range(1, 40): 
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(Xtrain, ytrain)
    pred_i = knn_i.predict(Xtest)
    error.append(metrics.mean_squared_error(ytest, pred_i))
knn_mse = min(error)
k_value = error.index(knn_mse) + 1 #adding one since it is an index
print("KNN K Value: ", k_value)
print("KNN MSE: ", knn_mse)
# below is copied from the link above, it shows a really nice plot for the 
# different K values 
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')"""

#OLS - Classification
""" Xa = sm.add_constant(Xtrain)
Xb = sm.add_constant(Xtest)
ols = sm.OLS(ytrain, Xa).fit()
print(ols.summary())
ypred = ols.predict(Xb)
mse = metrics.mean_squared_error(ytest, ypred)
print("OLS MSE: ", mse)  """
#With classification we also seem to yield much smaller MSEs and greater accuracy scores, so we can use more classification techniques

#Random forest classification
""" rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(Xtrain,ytrain)
ypred = rfc.predict(Xtest)
mse = metrics.mean_squared_error(ytest, ypred)
print("Random Forest Test MSE: ", mse)
print("Accuracy (100 Trees):", metrics.accuracy_score(ytest, ypred))

importance = rfc.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance, tick_label=X.columns.tolist())
plt.xticks(rotation=45, ha='right')
plt.title('Red Wine Random Forest Feature Importances')
plt.show() """

#Decision tree classifiers

#Cross val to determine optimal depth of tree
""" mse_scores = pd.Series()
for i in range(1,13):
    dtc = tree.DecisionTreeClassifier(random_state=6, max_depth=i)
    dtc.fit(Xtrain, ytrain)
    ypred = dtc.predict(Xtest)
    mse = metrics.mean_squared_error(ytest, ypred)
    mse_scores = mse_scores.append(pd.Series([mse]))
    print("Test MSE for depth " + str(i) + ": " + str(mse))
plt.xlabel('Tree Depth')
plt.ylabel('Test MSE')
plt.plot(np.arange(1, 13), mse_scores) """
#Optimal depth looks to be 2

""" dtc = tree.DecisionTreeClassifier(max_depth=2, random_state=7)
dtc.fit(Xtrain, ytrain)
ypred = dtc.predict(Xtest)
mse = metrics.mean_squared_error(ytest, ypred)
print("Decision Tree Classifier Test MSE (Depth 2): ", mse)
print(dtc.get_depth())
tree.plot_tree(dtc)
dot_data = tree.export_graphviz(dtc, out_file='redWineTree.dot') """
