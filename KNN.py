from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


dataset=load_iris()

X_train,X_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.2,random_state=30)

print("Training Labels: ",y_train)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

for i in range(len(X_test)):
    x=X_test[i]
    x_new=np.array([x])
    y_pred=model.predict(x_new)

    print(f"Target: {dataset.target_names[y_test[i]]}, Predicted: {dataset.target_names[y_pred]}")


accuracy=model.score(X_test,y_test)
print("Accuracy Score: ",accuracy)
