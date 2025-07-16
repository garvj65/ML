from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score

dataset=pd.read_csv('tennis.csv')
X=dataset.iloc[:,0:-1].copy()
y=dataset.iloc[:,-1]

print("Data Before Encoding\n",dataset.head())
print("Features Before Encoding\n",X.head())
print("Target Before Encoding\n",y.head())


label_encoder={}

for column in X.columns:
    le=LabelEncoder()
    X[column]=le.fit_transform(X[column])
    label_encoder[column]=le

print("Features After Encoding\n",X.head())

target_encoder=LabelEncoder()
y=target_encoder.fit_transform(y)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
model=GaussianNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Actual Label: ",y_test)
print("Predicted Label: ",y_pred)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("Accuracy Score: ",accuracy_score(y_test,y_pred)*100)
