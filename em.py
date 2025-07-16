from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm
dataset=load_iris()

X=pd.DataFrame(dataset.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(dataset.target)
y.columns=['Targets']

print(X)
colormap=np.array(['red','lime','black'])
plt.figure(figsize=(14,7))
plt.subplot(1,3,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets],s=40)
plt.title("Real")

plt.subplot(1,3,2)
model=KMeans(n_clusters=3)
model.fit(X)
y_pred=np.choose(model.labels_,[0,1,2]).astype(np.int64)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y_pred],s=40)
plt.title("KMeans")

scaler=StandardScaler()
scaler.fit(X)

xsa=scaler.transform(X)
xs=pd.DataFrame(xsa,columns=X.columns)
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)

y_cluster_gmm=gmm.predict(xs)
plt.subplot(1,3,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y_cluster_gmm],s=40)
plt.title("GMM")

ari_kMeans=sm.adjusted_rand_score(y.Targets,y_pred)
ari_gmm=sm.adjusted_rand_score(y.Targets,y_cluster_gmm)
print(f"Adjusted Rank Index for K-Means: {ari_kMeans:.4f}")
print(f"Adjusted Rank Index for GMM: {ari_gmm:.4f}")

accuracy_kMeans=np.mean(y.Targets==y_pred)
accuracy_gmm=np.mean(y.Targets==y_cluster_gmm)
plt.show()
