import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
#import joblib

filename=r'C:\Users\ANUPMA\Desktop\Desktop\MLOps\WineQualityCheckFastApi\winequality-red.csv'
df=pd.read_csv(filename)
# print(df.head())
features=df.columns
#print(features)
X=df.drop(['quality'],axis=1)
y=df['quality']
#print(X.head())
#print(y.head())
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#joblib.dump(scaler, 'scaler.pkl')
with open ( "scaler.pkl", 'wb') as f:
    pickle.dump(scaler,f)
models=[RandomForestClassifier(), DecisionTreeClassifier()]
params=[ {
        'n_estimators' : [1, 10, 20, 100],
         'criterion' : ['gini', 'entropy', 'log_loss']
         },
        { 
        'criterion' : ['gini', 'entropy', 'log_loss'],
          'max_depth' : [1, 10, 20, 100] } 
          ]
acc=[]
for i in range(0,len(models)):
    svc=GridSearchCV(estimator=models[i],param_grid=params[i],cv=3,scoring='accuracy')
    svc.fit(X_train,y_train)
    best_model=models[i]
    best_params=svc.best_params_
    best_model.set_params(**best_params)
    best_model.fit(X_train,y_train)
    y_pred=best_model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    acc.append(accuracy)
print(acc)
best_model=models[acc.index(max(acc))]
pred_new=best_model.predict(X_test)
#accuracy=accuracy_score(y_test,pred_new)
