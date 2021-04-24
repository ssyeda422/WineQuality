import pandas as pd
import numpy as np
import seaborn as sns


wineRed = pd.read_csv(r'C:\Users\saniy\OneDrive\Documents\WPI 2020-2021\D TERM\MACHINE LEARNING\winequality-red.csv')
wineWhite = pd.read_csv(r'C:\Users\saniy\OneDrive\Documents\WPI 2020-2021\D TERM\MACHINE LEARNING\winequality-white.csv')

#Printing correlation matrix for red wine
print(wineRed.head())
sns.pairplot(wineRed) 
corrMatrix = wineRed.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#Printing correlation matrix for white wine
corrMatrix = wineWhite.corr()
sns.heatmap(corrMatrix, annot=True)
