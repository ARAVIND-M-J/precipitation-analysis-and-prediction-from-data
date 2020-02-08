import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r'C:\Users\MyPc\Desktop\ict\weatherAUS.csv')
data.columns
new=data.copy()
data.info()
data.shape
data.isna().sum()
data.isna().sum().sort_values(ascending=True)
145460*.14
data.head
missing=data.isna().sum()
missing>20364
missing[missing>20364].index
data.drop(missing[missing>20364].index,axis=1,inplace=True)
new
data.columns


sns.heatmap(data.corr(),annot=True)


data.corr()
data.drop(['Date', 'Location','RISK_MM'],axis=1,inplace=True)
data.columns
data

data.shape
data=data.drop_duplicates()
data.shape


data=data.dropna(how='any')
data.shape

from scipy import stats
z = np.abs(stats.zscore(data._get_numeric_data()))
print(z)
data=data[(z<3).all(axis=1)]
print(data.shape)

data.columns
data.dtypes
 data.corr
data
 
data['RainToday'].replace({'No':0,'Yes':1},inplace=True)
data['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)
data['RainTomorrow'].astype=int

data.dtypes
data['WindDir9am'].value_counts()
data['WindDir9am'].value_counts()

data.dtypes

# ONE HOT 3 COLUMNS ON SINGLE STEP
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
data= pd.get_dummies(data,columns=categorical_columns)
data.dtypes
data.iloc[4:8]

sns.heatmap(data.corr(),annot=True)
  

x=data.iloc[:,data.columns!='RainTomorrow']
y=data.iloc[:,data.columns=='RainTomorrow']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
    
#RANDOM FORRE
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
y_pred

from sklearn.metrics import *
print('confusion matrix is' ,confusion_matrix(y_test,y_pred))
print('Precision is',precision_score(y_test,y_pred))
print('recall is', recall_score(y_test,y_pred))
print('accuracy is',accuracy_score(y_test,y_pred))
print('f1 score is',f1_score(y_test,y_pred))
#accuracy is 0.843669849510213
#f1 score is 0.5260889929742387

#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_dtc=dtc.predict(X_test)
y_pred_dtc

from sklearn.metrics import *
print('confusion matrix is' ,confusion_matrix(y_test,y_pred_dtc))
print('Precision is',precision_score(y_test,y_pred_dtc))
print('recall is', recall_score(y_test,y_pred_dtc))
print('accuracy is',accuracy_score(y_test,y_pred_dtc))
print('f1 score is',f1_score(y_test,y_pred_dtc))
#accuracy is 0.7938259015481598
#f1 score is 0.5095560129373713

#LOGISTIC REGG
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
y_pred_log=log.predict(X_test)
y_pred_log

from sklearn.metrics import *
print('confusion matrix is' ,confusion_matrix(y_test,y_pred_log))
print('Precision is',precision_score(y_test,y_pred_log))
print('recall is', recall_score(y_test,y_pred_log))
print('accuracy is',accuracy_score(y_test,y_pred_log))
print('f1 score is',f1_score(y_test,y_pred_log))
#accuracy is 0.8524149439139705
#f1 score is 0.5740278273278631

data


from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred_knn=classifier.predict(X_test)














