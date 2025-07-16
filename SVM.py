import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X, y = X[y != 2, :2], y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Support Vectors: {len(model.support_vectors_)}")
print("Support Vectors:\n ", model.support_vectors_)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

xx, yy = np.meshgrid(np.arange(X[:,0].min()-1, X[:,0].max()+1, 0.02),
                     np.arange(X[:,1].min()-1, X[:,1].max()+1, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(scaler.transform(grid)).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
sv = scaler.inverse_transform(model.support_vectors_)
plt.scatter(sv[:,0], sv[:,1], s=100, facecolors='none', edgecolors='blue', label='Support Vectors')
plt.legend(); plt.title('Linear SVM'); plt.show()
