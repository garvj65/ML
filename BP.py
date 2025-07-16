import numpy as np

X=np.array([[2,9],[1,5],[3,6]])
y=np.array([[92],[86],[89]])


X=X/np.amax(X,axis=0) #Chances of Error
y=y/100
class NeuralNetwork:
    def __init__(self):
        self.input_size=2
        self.hidden_size=3
        self.output_size=1
        self.weights_input_hidden=np.random.rand(self.input_size,self.hidden_size)
        self.weights_hidden_output=np.random.rand(self.hidden_size,self.output_size)

    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))

    def sigmoid_prime(self,x):
        return x*(1-x)

    def forward(self,input_data):
        self.input_layer=input_data
        self.hidden_input=np.dot(self.input_layer,self.weights_input_hidden)
        self.hidden_output=self.sigmoid(self.hidden_input)

        self.final_input=np.dot(self.hidden_output,self.weights_hidden_output)
        self.predicted_output=self.sigmoid(self.final_input)

        return self.predicted_output

    def backward(self,target_output):
        error=target_output-self.predicted_output
        output_delta=error*self.sigmoid_prime(self.predicted_output)

        hidden_error=output_delta.dot(self.weights_hidden_output.T)
        hidden_delta=hidden_error*self.sigmoid_prime(self.hidden_output) #Chances of Error

        self.weights_input_hidden+=self.input_layer.T.dot(hidden_delta) 
        self.weights_hidden_output+=self.hidden_output.T.dot(output_delta)

    def train(self,X,y):
        self.forward(X)
        self.backward(y)

NN=NeuralNetwork()

print("Input: \n",X)
print("Predicted Output: ",NN.forward(X))
print("Loss Before Training:\n ",np.mean(np.square(y-NN.forward(X))))


NN.train(X,y)
print("Predicted Output: \n",NN.forward(X))
print("Loss After Training:\n ",np.mean(np.square(y-NN.forward(X))))
