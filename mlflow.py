import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.model_selection import train_test_split,GridSearchCV
import joblib
from sklearn.metrics import accuracy_score


file=r'/workspaces/MLOps_E2E_1st/winequality-red.csv'
df=pd.read_csv(file)
#print(df.head())

#Seperating the imput and output
X=df.drop(['quality'],axis=1)
Y=df['quality']

#print(X,Y)
#Splitting and scaling the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
joblib.dump(scaler,'scaler.pkl')
#print(X_train,X_test)

#best model selection and model training 

model_list=[RandomForestClassifier(),DecisionTreeClassifier()]
best_models={}
params={
    #'n_estimators': [10,35,50,100],
    'max_depth': [None,5,10,20],
    'criterion': ['gini','entropy'],
    'random_state': [42]
}
for model in model_list:
    gridsearch=GridSearchCV(model,params,cv=3,scoring='accuracy')
    gridsearch.fit(X_train,Y_train)
#   print(model,gridsearch.best_params_)
    #model.set_params(**gridsearch.best_params_)
    #model.fit(X_train,Y_train)
    model=gridsearch.best_estimator_
# we can use gridsearch.best_estimators_ as well, then we can remove the set_params and fit()
    Y_train_pred=model.predict(X_train)
    Y_test_pred=model.predict(X_test)
    Accuracy_train=accuracy_score(Y_train,Y_train_pred)
    Accuracy=accuracy_score(Y_test,Y_test_pred)
    best_models[model]=Accuracy


#print(max(best_model.values()))
best_model=max(best_models,key=best_models.get)
best_model_name=type(best_model).__name__

joblib.dump(best_model,f'{best_model_name}.pkl')


    
    






