# coding: utf-8


import numpy as np
import sys
import time



def sigmoid(x):
    ret =  1.0 / (1.0 + np.exp(-x))
    
    return ret

def dsigmoid(x):
    y = sigmoid(x)
    return x * (1.0 - x)

#def softmax(x):
#    ret = np.zeros( x.shape )
#    temp = np.exp(x)
#    temp_sum = np.sum(temp , axis=1)
#    for i in range(x.shape[0]):
#        ret[i] = temp[i,:] / temp_sum[i]
#   return ret

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2




class DeepNeuralNet:

    def __init__(self, n_inputs, n_hidden1 , n_hidden2 , n_hidden3,   n_outputs):
        """
        Initializing Parameter
        5 layers
        """
        numpy_rng = np.random.RandomState(1234)
        self.W1 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(n_inputs , n_hidden1)))
        self.h1_bias = np.zeros(n_hidden1, dtype=float)
        self.W2 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(n_hidden1 , n_hidden2)))
        self.h2_bias = np.zeros(n_hidden2 , dtype=float)
        self.W3 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(n_hidden2 , n_hidden3)))
        self.h3_bias = np.zeros(n_hidden3 , dtype=float)
        self.W4 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(n_hidden3 , n_outputs)))
        self.out_bias = np.zeros(n_outputs, dtype=float)
        self.n_outputs = n_outputs
        self.cost = []
        
    def forward(self, x ):
        """
        Execute feed forward computation
        """
        z1 = sigmoid(np.dot(x , self.W1) + self.h1_bias)
        z2 = sigmoid(np.dot(z1 , self.W2) + self.h2_bias)
        z3 = sigmoid(np.dot(z2 , self.W3) + self.h3_bias)
        y = softmax(np.dot(z3 , self.W4) + self.out_bias)
        return y

    def get_cost(self, data, target):
        """
        Cost Function : Cross Entropy
        """
        #ret = target*(np.log(y))
        #return  -np.sum(ret)
        N = data.shape[0]
        cross_entropy = 0.
        for i in range(N):
            y = self.forward(data[i])
            d = target[i]
            cross_entropy += np.sum( (d * np.log(y)) )
        return -cross_entropy
    
    def cost(self, data, target):
        """
        最小化したい誤差関数
        """
        N = data.shape[0]
        E = 0.0
        for i in range(N):
            y, t = self.forward(data[i]), target[i]
            E += np.sum((y - t) * (y - t))
        return 0.5 * E / float(N)
        

    def train(self, data, target, epochs=15, learning_rate=0.1 , monitor=2000):

        """
        Stochastic Gradient Decent (SGD) : 1 Sample in training data
        """
        start = time.time()
        
        for epoch in range(epochs):
            idx = np.random.randint(0 , data.shape[0])#np.random.permutation(N)
            x  , d = data[idx] , target[idx]
            # calculating Actual Data
            z1 = sigmoid(np.dot(x , self.W1) + self.h1_bias)
            z2 = sigmoid(np.dot(z1 , self.W2) + self.h2_bias)
            z3 = sigmoid(np.dot(z2 , self.W3) + self.h3_bias)
            y = sigmoid(np.dot(z3 , self.W4) + self.out_bias)
            # Hidden Layer3 - Output Layer
            out_delta = (y - d) * dsigmoid(y)
            grad_W4 = np.dot(np.atleast_2d(z3).T, np.atleast_2d(out_delta))
            grad_out_bias = out_delta
            self.W4 -= learning_rate * grad_W4
            self.out_bias -= learning_rate * grad_out_bias
            # Hidden Layer2 - Hidden Layer3   
            h3_delta = np.dot(self.W4 , out_delta) * dsigmoid(z3)
            grad_W3 = np.dot(np.atleast_2d(z2).T , np.atleast_2d(h3_delta))
            grad_h3_bias = h3_delta
            self.W3 -= learning_rate * grad_W3
            self.h3_bias -= learning_rate * grad_h3_bias
            # Hidden Layer1 - Hidden Layer2   
            h2_delta = np.dot(self.W3 , h3_delta) * dsigmoid(z2)
            grad_W2 = np.dot(np.atleast_2d(z1).T , np.atleast_2d(h2_delta))
            grad_h2_bias = h2_delta
            self.W2 -= learning_rate * grad_W2
            self.h2_bias -= learning_rate * grad_h2_bias
            # Input Layer - Hidden Layer1   
            h1_delta = np.dot(self.W2 , h2_delta) * dsigmoid(z1)
            grad_W1 = np.dot(np.atleast_2d(x).T , np.atleast_2d(h1_delta))
            grad_h1_bias = h1_delta
            self.W1 -= learning_rate * grad_W1
            self.h1_bias -= learning_rate * grad_h1_bias
            # 現在の目的関数の値を出力
            if monitor != None and epoch % monitor == 0:
                cost = self.get_cost(data, target)
                print("Epoch: {0}, Cost: {1}").format(epoch, cost)
                self.cost.append(cost)
        self.elapsed_time = time.time() - start
        print "学習時間 : {0}".format(self.elapsed_time) + "[sec]"
        
        
    def predict(self , x):
        return np.argmax(self.forward(x))
    
    
    
            

    def accuracy(self, x_, y_):
        """
        accuracy validation set
        """
        d_ = self.forward(x_)
        d_hat = np.argmax(d_,axis=1)
        return (d_hat == y_).mean()
        
if __name__ == "__main__":
    import loadMNIST

    #XOR Test
    inputs = np.array([[0, 1], [0 ,0], [1, 0], [1, 1]])
    targets = np.array([1,0,1,0])
    nn = NeuralNet(2,2,2)
    nn.train(inputs , targets , epochs=30000, learning_rate=0.1, batch_size=1, momentum_rate=0.5)
    #nn.ex_train(inputs , targets , learning_rate=0.1 ,  epoches=50000 , monitor_period=2000)
    print nn.forward(inputs) , np.argmax(nn.forward(inputs) , axis=1)

    #MNIST Test
    #inputs = loadMNIST.loadInputData()
    #input_data = inputs["data"] / 255.0
    #nn = NeuralNet(784 , 200 , 10)
    #nn.train(input_data , inputs["target"] , epochs=100)
    
