#coding:utf-8


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




class DeepBeliefNet10:

	def __init__(self, inputs , Learn_flag=True , PreTrain_flag=True):
        """
        Pretraing Parameter
        5 layers
        """
        numpy_rng = np.random.RandomState(1234)
        self.pre_elapsed_time = 0
        if PreTrain == True:
	        if Learn_flag == True:
    	        print "1-2層目の事前学習を行います..."
	            rbm1 = RestrictedBM.RBM(inputs, n_hidden=50)
	            rbm1.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h1_samples = rbm1.get_samples(inputs)
	            print "2-3層目の事前学習を行います..."
	            rbm2 = RestrictedBM.RBM(h1_samples , n_hidden=50)
	            rbm2.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h2_samples = rbm2.get_samples(h1_samples)
	            print "3-4層目の事前学習を行います..."
	            rbm3 = RestrictedBM.RBM(h2_samples , n_hidden=50)
	            rbm3.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h3_samples = rbm3.get_samples(h2_samples)
	            print "4-5層目の事前学習を行います..."
	            rbm4 = RestrictedBM.RBM(h3_samples , n_hidden=50)
	            rbm4.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h4_samples = rbm4.get_samples(h3_samples)
	            print "5-6層目の事前学習を行います..."
	            rbm5 = RestrictedBM.RBM(h4_samples , n_hidden=50)
	            rbm5.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h5_samples = rbm5.get_samples(h4_samples)
	            print "6-7層目の事前学習を行います..."
	            rbm6 = RestrictedBM.RBM(h5_samples , n_hidden=50)
	            rbm6.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h6_samples = rbm6.get_samples(h5_samples)
	            print "7-8層目の事前学習を行います..."
	            rbm7 = RestrictedBM.RBM(h6_samples , n_hidden=50)
	            rbm7.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            h7_samples = rbm7.get_samples(h6_samples)
	            print "8-9層目の事前学習を行います..."
	            rbm8 = RestrictedBM.RBM(h7_samples , n_hidden=50)
	            rbm8.train(learning_rate=0.05, k=1, epochs=5000, batch_size=100, monitor=1000)
	            self.W1 = rbm1.W
	            self.h1_bias = rbm1.hbias
	            self.W2 = rbm2.W
	            self.h2_bias = rbm2.hbias
	            self.W3 = rbm3.W
	            self.h3_bias = rbm3.hbias
	            self.W4 = rbm4.W
	            self.h4_bias = rbm4.hbias
	            self.W5 = rbm5.W
	            self.h5_bias = rbm5.hbias
	            self.W6 = rbm6.W
	            self.h6_bias = rbm6.hbias
	            self.W7 = rbm7.W
	            self.h7_bias = rbm7.hbias
	            self.W8 = rbm8.W
	            self.h8_bias = rbm8.hbias
	            
	            print "パラメータを保存します..."
	            params = {"W":rbm1.W , "vbias":rbm1.vbias , "hbias":rbm1.hbias , "elapsed_time":rbm1.elapsed_time}
	            f = open("10rbm1.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm2.W , "vbias":rbm2.vbias , "hbias":rbm2.hbias , "elapsed_time":rbm2.elapsed_time}
	            f = open("rbm2.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm3.W , "vbias":rbm3.vbias , "hbias":rbm3.hbias , "elapsed_time":rbm3.elapsed_time}
	            f = open("10rbm3.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm4.W , "vbias":rbm4.vbias , "hbias":rbm4.hbias , "elapsed_time":rbm4.elapsed_time}
	            f = open("10rbm4.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm5.W , "vbias":rbm5.vbias , "hbias":rbm5.hbias , "elapsed_time":rbm5.elapsed_time}
	            f = open("10rbm5.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm6.W , "vbias":rbm6.vbias , "hbias":rbm6.hbias , "elapsed_time":rbm6.elapsed_time}
	            f = open("10rbm6.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm7.W , "vbias":rbm7.vbias , "hbias":rbm7.hbias , "elapsed_time":rbm7.elapsed_time}
	            f = open("10rbm7.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            params = {"W":rbm8.W , "vbias":rbm8.vbias , "hbias":rbm8.hbias , "elapsed_time":rbm8.elapsed_time}
	            f = open("10rbm8.pkl" , "w")    
	            cPickle.dump(params,f)
	            f.close
	            
	            self.pre_elapsed_time = rbm1.elapsed_time + rbm2.elapsed_time + rbm3.elapsed_time
	        else:
	            print "パラメータを読み込みます..."
	            f = open("10rbm1.pkl")
	            params = cPickle.load(f)
	            self.W1 = params["W"]
	            self.h1_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm2.pkl")
	            params = cPickle.load(f)
	            self.W2 = params["W"]
	            self.h2_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm3.pkl")
	            params = cPickle.load(f)
	            self.W3 = params["W"]
	            self.h3_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm4.pkl")
	            params = cPickle.load(f)
	            self.W4 = params["W"]
	            self.h4_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm5.pkl")
	            params = cPickle.load(f)
	            self.W5 = params["W"]
	            self.h5_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm6.pkl")
	            params = cPickle.load(f)
	            self.W6 = params["W"]
	            self.h6_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm7.pkl")
	            params = cPickle.load(f)
	            self.W7 = params["W"]
	            self.h7_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            f = open("10rbm8.pkl")
	            params = cPickle.load(f)
	            self.W8 = params["W"]
	            self.h8_bias = params["hbias"]
	            self.pre_elapsed_time += params["elapsed_time"]
	            f.close()
	            
	        
	        self.W9 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 10)))
	        self.out_bias = np.zeros(10, dtype=float)
	        self.cost = []
	        #self.n_outputs = n_outputs

		else :   
	        self.W1 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(784 , 50)))
	        self.h1_bias = np.zeros(n_hidden1, dtype=float)
	        self.W2 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h2_bias = np.zeros(n_hidden2 , dtype=float)
	        self.W3 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h3_bias = np.zeros(n_hidden3 , dtype=float)
	        self.W4 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h4_bias = np.zeros(n_hidden4 , dtype=float)
	        self.W5 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h5_bias = np.zeros(n_hidden5 , dtype=float)
	        self.W6 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h6_bias = np.zeros(n_hidden6 , dtype=float)
	        self.W7 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h7_bias = np.zeros(n_hidden7 , dtype=float)
	        self.W8 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 50)))
	        self.h8_bias = np.zeros(n_hidden8 , dtype=float)
	        self.W9 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(50 , 10)))
	        self.out_bias = np.zeros(10, dtype=float)
	        self.cost = []
        
    def forward(self, x ):
        """
        Execute feed forward computation
        """
        z1 = sigmoid(np.dot(x , self.W1) + self.h1_bias)
        z2 = sigmoid(np.dot(z1 , self.W2) + self.h2_bias)
        z3 = sigmoid(np.dot(z2 , self.W3) + self.h3_bias)
        z4 = sigmoid(np.dot(z3 , self.W4) + self.h4_bias)
        z5 = sigmoid(np.dot(z4 , self.W5) + self.h5_bias)
        z6 = sigmoid(np.dot(z5 , self.W6) + self.h6_bias)
        z7 = sigmoid(np.dot(z6 , self.W7) + self.h7_bias)
        z8 = sigmoid(np.dot(z7 , self.W8) + self.h8_bias)
        
        y = softmax(np.dot(z3 , self.W9) + self.out_bias)
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
            z4 = sigmoid(np.dot(z3 , self.W4) + self.h4_bias)
            z5 = sigmoid(np.dot(z4 , self.W5) + self.h5_bias)
            z6 = sigmoid(np.dot(z5 , self.W6) + self.h6_bias)
            z7 = sigmoid(np.dot(z6 , self.W7) + self.h7_bias)
            z8 = sigmoid(np.dot(z7 , self.W8) + self.h8_bias)
            
            y = sigmoid(np.dot(z3 , self.W9) + self.out_bias)
            # Hidden Layer8 - Output Layer
            out_delta = (y - d) * dsigmoid(y)
            grad_W9 = np.dot(np.atleast_2d(z8).T, np.atleast_2d(out_delta))
            grad_out_bias = out_delta
            self.W9 -= learning_rate * grad_W9
            self.out_bias -= learning_rate * grad_out_bias
            
            # Hidden Layer7 - Hidden Layer8   
            h8_delta = np.dot(self.W8 , out_delta) * dsigmoid(z8)
            grad_W8 = np.dot(np.atleast_2d(z7).T , np.atleast_2d(h8_delta))
            grad_h8_bias = h8_delta
            self.W8 -= learning_rate * grad_W8
            self.h8_bias -= learning_rate * grad_h8_bias
            
            # Hidden Layer6 - Hidden Layer7   
            h7_delta = np.dot(self.W7 , h8_delta) * dsigmoid(z7)
            grad_W7 = np.dot(np.atleast_2d(z6).T , np.atleast_2d(h7_delta))
            grad_h7_bias = h7_delta
            self.W7 -= learning_rate * grad_W7
            self.h7_bias -= learning_rate * grad_h7_bias
            
            # Hidden Layer5 - Hidden Layer6   
            h6_delta = np.dot(self.W6 , h7_delta) * dsigmoid(z6)
            grad_W6 = np.dot(np.atleast_2d(z5).T , np.atleast_2d(h6_delta))
            grad_h6_bias = h6_delta
            self.W6 -= learning_rate * grad_W6
            self.h6_bias -= learning_rate * grad_h6_bias
            
            # Hidden Layer4 - Hidden Layer5   
            h5_delta = np.dot(self.W5 , h6_delta) * dsigmoid(z5)
            grad_W5 = np.dot(np.atleast_2d(z4).T , np.atleast_2d(h5_delta))
            grad_h5_bias = h5_delta
            self.W5 -= learning_rate * grad_W5
            self.h5_bias -= learning_rate * grad_h5_bias
            
            # Hidden Layer3 - Hidden Layer4   
            h4_delta = np.dot(self.W4 , h5_delta) * dsigmoid(z4)
            grad_W4 = np.dot(np.atleast_2d(z3).T , np.atleast_2d(h4_delta))
            grad_h4_bias = h4_delta
            self.W4 -= learning_rate * grad_W4
            self.h4_bias -= learning_rate * grad_h4_bias
            
            # Hidden Layer2 - Hidden Layer3   
            h3_delta = np.dot(self.W3 , h4_delta) * dsigmoid(z3)
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
    
