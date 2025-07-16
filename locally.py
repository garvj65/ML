import numpy as np
import matplotlib.pyplot as plt
X=np.array([1,2,3,4,5])
y=np.array([1,2,1.3,3.75,2.25])

def gaussian_weights(X_query,X,y,tau):
    return np.exp(-((X-X_query)**2)/(2*tau**2))

def lwr(X_query,X,y,tau):
    Weights=gaussian_weights(X_query,X,y,tau)
    W=np.diag(Weights)
    x_mat=np.vstack([np.ones_like(X),X]).T
    theta=np.linalg.pinv(x_mat.T @ W @x_mat) @(x_mat.T @W @y)
    return np.array([1,X_query]) @theta,theta

X_query=3
tau=1.0
y_pred,theta=lwr(X_query,X,y,tau)

print("Observed value at x=3: ",y[X==X_query][0])
print(f"Predicted value at x=3: {y_pred:3f}")
print(f"Coefficients: Intercepts={theta[0]:.3f}, Slope={theta[1]:.3f}")

x_vals=np.linspace(1,5,100)
y_vals=[lwr(x,X,y,tau)[0] for x in x_vals]

plt.scatter(X,y,color='red',label='Data Points')
plt.plot(x_vals,y_vals,color='blue',label='LWR Prediction')
plt.scatter(X_query,y_pred,color='green',label='Predicted Value at x=3')
plt.scatter(X_query,y[X==X_query][0],color='orange',label='Observed value at x=3')
plt.legend()
plt.show()
