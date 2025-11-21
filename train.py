import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import pickle
import numpy as np
#import joblib

filename=r'/workspaces/MLOps_E2E_1st/winequality-red.csv'
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
accuracy=accuracy_score(y_test,pred_new)
metrics={
    "Accuracy": accuracy
#####Belows are only for binary classfications,here we have dataset (winequality-red.csv) has multi-class labels in quality (e.g., values like 3,4,5,6,7,8).
    # "Precision": precision_score(y_test,pred_new),
    # "Recall": recall_score(y_test,pred_new),
    # "F1-Score": f1_score(y_test,pred_new),
    # "ROC-AUC": roc_auc_score(y_test,pred_new)
}
import os
import dagshub
os.environ["DAGSHUB_USERNAME"] = "ap1305"
os.environ["DAGSHUB_KEY"] = "00c665a88028b39a82053151d6d04df5ff7a902f"


#dagshub.clear_token_cache()
# After running the above line, your next DagsHub operation will re-authenticate your connection.

# Now, your project will connect using the correct account [1].

# from dagshub.auth import clear_token_cache
# clear_token_cache() 
dagshub.init(repo_name='MLOps_E2E_1st',repo_owner='ap1305',mlflow=True)
mlflow.set_experiment("wineQualityCheck")
with mlflow.start_run():
    mlflow.set_tag("Author","Anish")
    mlflow.log_metric("accuracy",float(accuracy))
    mlflow.log_metrics(metrics)
    mlflow.log_params(params=best_params)
    model_name="best_model.pkl"
    with open(model_name,"wb") as f:
        pickle.dump(best_model,f)
    mlflow.log_artifact(model_name,"best_model")