import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle

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
X_test=scaler.fit(X_test)
joblib.dump(scaler, 'scaler.pkl')