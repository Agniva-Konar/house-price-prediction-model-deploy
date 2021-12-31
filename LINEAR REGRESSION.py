import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

data=pd.read_excel(r'C:\Users\AGNIVA\Desktop\MSC PROJECTS\1ST SEM\HOUSE PRICE PREDICTION.xlsx')
data.head()

data.columns

data.drop(data.columns[[0,2,4,7,9,11,13,15,16,17,19,21,25,26,27,29,31,33,35,37,39,41,43,45,47,49,52,56,67,70,73,75,77,78,82,89,91,92,93,95]],axis=1,inplace=True)
data.head()

np.shape(data)

data.dtypes

print(data.isnull().values.any())
data.dropna(inplace=True)
np.shape(data)

X = data.iloc[:, :57]
Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=5)

X_train.index=list(range(0,960))
Y_train.index=list(range(0,960))
X_test.index=list(range(0,412))
Y_test.index=list(range(0,412))

import statsmodels.api as sm
from statsmodels.formula.api import ols
model=sm.OLS(Y_train,X_train).fit()
model.summary()

influence=model.get_influence()
influence.summary_frame().head() #dataframe showing all the influence diagnostics

cooksdist=influence.cooks_distance[0]
cook=cooksdist[cooksdist>(4/960)]
influential=np.where(np.isin(cooksdist,cook))
influential=np.asarray(influential)
print(influential) #potential influential observations

studres=influence.resid_studentized_external
s=studres[abs(studres)>3]
outliers=np.where(np.isin(studres,s))
outliers=np.asarray(outliers)
print(outliers) #potential outliers

hatvalues=influence.hat_matrix_diag
hat=hatvalues[hatvalues>((3*57)/960)]
highlev=np.where(np.isin(hatvalues,hat))
highlev=np.asarray(highlev)
print(highlev) #potential high leverage values

commonvals=np.intersect1d(influential,highlev)
commonvals

X_train.drop(commonvals,axis=0,inplace=True)
Y_train.drop(commonvals,axis=0,inplace=True)

from sklearn.linear_model import LinearRegression
model_new=LinearRegression()
model_new.fit(X_train,Y_train)

pickle.dump(model_new,open('LIN_REG.pkl','wb'))

