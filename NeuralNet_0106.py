# coding: utf-8


import numpy as np
import sys


#デバッグ関数
def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

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




class NeuralNet:

    def __init__(self, n_inputs, n_hidden1,  n_outputs):
        """
        Initializing Parameter
        3 layers
        """
        numpy_rng = np.random.RandomState(1234)
        self.W1 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(n_inputs , n_hidden1)))
        #self.W1 = np.array([[0.1,-0.1],[0.1,-0.1]]) #For Debug
        self.h1_bias = np.zeros(n_hidden1, dtype=float)
        self.W2 = np.array(numpy_rng.uniform(low=-1.0 , high=1.0 , size=(n_hidden1 , n_outputs)))
        #self.W2 = np.array([[0.5,0.5],[0.5,0.5]]) #For Debug
        self.out_bias = np.zeros(n_outputs, dtype=float)
        
    def forward(self, x ):
        """
        Execute feed forward computation
        """
        z1 = sigmoid(np.dot(x , self.W1) + self.h1_bias)
        #set_trace()
        y = softmax(np.dot(z1 , self.W2) + self.out_bias)
        return y

    def get_cost(self, data, target):
        """
        Cost Function : Cross Entropy
        """
        y = self.forward(data)
        #ret = target*(np.log(y))
        #return  -np.sum(ret)
        cross_entropy = - np.mean(
            np.sum(target * np.log(y) +
            (1 - target) * np.log(1 - y),
                      axis=1))

        return cross_entropy
    
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
        

    def train(self, data_org, target_org, epochs=15, learning_rate=0.1,batch_size=100, momentum_rate=0.5):

        """
        Stochastic Gradient Decent (SGD) , and minibatch
        """
        #Alias
        W1 = self.W1
        W2 = self.W2
        h1_bias = self.h1_bias
        out_bias = self.out_bias
        N = 4#70000 #training sample size
        TotalSize = data_org.shape[0] #the number of all training samples
        data , data_ = data_org , data_org#np.split(data_org , [N])
        target , target_ = target_org , target_org#np.split(target_org , [N])
        dTarget = self.ConvertVector2Matrix(target , N)
        #dTarget_ = self.ConvertVector2Matrix(target_ , TotalSize-N)
        for epoch in range(epochs):
            # Random minibatch Sampling
            idxs = [0,1,2,3]#np.random.permutation(N)
            for i in xrange(0 , N , batch_size):
                momentum_term1=0 #initialization momentum
                momentum_term2=0
                x, d_tmp = data[idxs[i:i+batch_size]], target[idxs[i:i+batch_size]]
                # caluculating Acutual Data
                d = self.ConvertVector2Matrix(d_tmp , batch_size)
                z1 = sigmoid(np.dot(x , W1) + h1_bias)
                y = softmax(np.dot(z1 , W2) + out_bias)
                # Delta
                out_delta = d - y
                h1_delta = dsigmoid(z1) * (np.dot(out_delta , W2.T))
                in_delta = dsigmoid(x) * (np.dot(h1_delta , W1.T))
                # gradients
                grad_W2 = np.dot(out_delta.T , z1) / batch_size
                grad_W1 = np.dot(h1_delta.T , x) / batch_size
                grad_out_bias = np.sum(out_delta , axis=0) / batch_size
                grad_h1_bias = np.sum(h1_delta , axis=0) / batch_size
                # Variation :
                Vari_W2 = - learning_rate * grad_W2 + momentum_rate * momentum_term1
                Vari_W1 = - learning_rate * grad_W1 + momentum_rate * momentum_term2
                Vari_out_bias = - learning_rate * grad_out_bias
                Vari_h1_bias = - learning_rate * grad_h1_bias
                # Update :
                self.W2 += Vari_W2
                self.W1 += Vari_W1
                self.out_bias += Vari_out_bias
                self.h1_bias += Vari_h1_bias
                #Upgrade momentum term
                momentum_term1 = 0#Vari_W1
                momentum_term2 = 0#Vari_W2
            # Cost : Cross Entropy
            cost = self.get_cost(data , dTarget)
            # caluculating validation error
            acc = self.accuracy(data_ , target_)
            #acc = self.get_cost(data_ , dTarget_)
            print("Epoch: {0}, Cost: {1}, Acc: {2} , Weight:\n {3} ,\n {4}").format(epoch, cost, acc , self.W2,self.W1)
        print("Training finished")
        
    def ex_train(self, data, target, epoches=30000, learning_rate=0.1,\
              monitor_period=None):
        """
        Stochastic Gradient Decent (SGD) による学習
        """
        for epoch in range(epoches):
            # 学習データから1サンプルをランダムに選ぶ
            index = np.random.randint(0, data.shape[0])
            x, t = data[index], target[index]

            # 入力から出力まで前向きに信号を伝搬
            h = sigmoid(np.dot(self.W1.T, x) + self.h1_bias)
            y = sigmoid(np.dot(self.W2.T, h) + self.out_bias)

            # 隠れ層->出力層の重みの修正量を計算
            output_delta = (y - t) * dsigmoid(y)
            grad_W2 = np.dot(np.atleast_2d(h).T, np.atleast_2d(output_delta))

            # 隠れ層->出力層の重みを更新
            self.W2 -= learning_rate * grad_W2
            self.out_bias -= learning_rate * output_delta

            # 入力層->隠れ層の重みの修正量を計算
            hidden_delta = np.dot(self.W2, output_delta) * dsigmoid(h)
            grad_W1 = np.dot(np.atleast_2d(x).T, np.atleast_2d(hidden_delta))
            
            # 入力層->隠れ層の重みを更新
            self.W1 -= learning_rate * grad_W1
            self.h1_bias -= learning_rate * hidden_delta

            # 現在の目的関数の値を出力
            if monitor_period != None and epoch % monitor_period == 0:
                print "Epoch %d, Cost %f" % (epoch, self.cost(data, target))

        print "Training finished."
        
            

    def accuracy(self, x_, y_):
        """
        accuracy validation set
        """
        d_ = self.forward(x_)
        d_hat = np.argmax(d_,axis=1)
        return (d_hat == y_).mean()

    def ConvertVector2Matrix(self,x , size):
        # return d converted in (size , 10)
        d = np.zeros((size,2))#np.zeros((size,10))
        for i in range(size):
            d[i , x[i] ]  = 1
        return d

        


if __name__ == "__main__":
    import loadMNIST

    #XOR Test
    inputs = np.array([[0, 1], [0 ,0], [1, 0], [1, 1]])
    targets = np.array([1,0,1,0])
    nn = NeuralNet(2,2,2)
    nn.ex_train(inputs , targets , learning_rate=0.1 ,  epoches=50000 , monitor_period=2000)
    print nn.forward(inputs) , np.argmax(nn.forward(inputs) , axis=1)

    #MNIST Test
    #inputs = loadMNIST.loadInputData()
    #input_data = inputs["data"] / 255.0
    #nn = NeuralNet(784 , 200 , 10)
    #nn.train(input_data , inputs["target"] , epochs=100)
    
