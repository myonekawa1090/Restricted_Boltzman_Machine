#coding: utf-8

import sys
import numpy as np
import pickle
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import time
from sklearn.learning_curve import learning_curve

def sigmoid(x):
    ret = 1.0 / (1.0 + np.exp(-x))

    return ret 

def loadInputData():
    f = open("binary_mnist.pkl" , "r")
    input = pickle.load(f)
    f.close()

    return input

def readInputData(test_size=0.1):
    
    print u"mnist��ǂݍ���ł��܂�..."
    mnist = datasets.fetch_mldata("MNIST original")
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=test_size)
    input = X_test
    #set_trace()
    
    print "mnist���Q�l�����Ă��܂�..."
    j_size = input[:,0].size
    i_size = input[0].size
    for j in range(j_size):
        for i in range(i_size):
            if input[j , i] >= 128:
                input[j , i] = 1
            else:
                input[j,i] = 0
    
    print "binary_mnist.pkl�ɕۑ����Ă��܂�..."
    f = open("binary_mnist.pkl" , "w+")
    pickle.dump(input , f)
    f.close()
                
    return input

def writeParameterToPKL(Data , Filename="RBMParam.pkl"):
    f = open(Filename , "w")
    pickle.dump(Data , f)

    return "�������݂܂���."

def loadParameterToPkl(Filename="RBMParam.pkl"):
    f = open(Filename , "r")
    return pickle.load(f)
    

class RBM(object):
    def __init__(self, input=None  , n_hidden=100,
                 W=None , hbias=None , vbias=None, numpy_rng=None):
        self.n_hidden = n_hidden # num of units in hidden units layer
        if input is None:
            input = np.array([[1,2,3,4,5],[10,11,12,13,14,15]])
        self.input = input
        
        n_visible = input.shape[1]
        self.n_visible = n_visible
        
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        
        if W is None:
            #a = 1. / n_visible
            initial_W = np.array(numpy_rng.normal(0,0.01,size=(n_visible , n_hidden)))
            W = initial_W
            
     
        if vbias is None:
            vbias = np.zeros(n_visible)
            #rate_1_input = np.sum(input , axis=0)/float(input.shape[0])
            #vbias = np.log(rate_1_input / (1 - rate_1_input))

        if hbias is None:
            hbias = np.zeros(n_hidden) # �B��o�C�A�X������
           

        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
       # set_trace()
    def contrastive_divergence(self,input , k=1):
        
        v0_samples  = input
        pre_sigmoid_activation = np.dot(v0_samples , self.W) + self.hbias
        h0_probability = sigmoid(pre_sigmoid_activation)
        h0_samples = self.numpy_rng.binomial(size=h0_probability.shape , n=1 , p = h0_probability)

        for step in range(k):
            if step == 0:
                hk_samples , vk_samples , hk_probability , vk_probability = self.gibbs_sampling(v0_samples)
            else:
                hk_samples , vk_samples , hk_probability , vk_probability = self.gibbs_sampling(vk_samples)
                    
                #chain end = nv_samples
                    

        dw = (np.dot(v0_samples.T , h0_probability) - np.dot(vk_samples.T , hk_probability))
        dv = np.sum(v0_samples , axis=0) - np.sum(vk_samples , axis=0)
        dh = np.sum(h0_probability , axis=0) - np.sum(hk_probability , axis=0)

        return [dw , dv , dh]

   
    def gibbs_sampling(self , v0_samples):
        pre_sigmoid_activation = np.dot(v0_samples , self.W) + self.hbias
        h0_probability = sigmoid(pre_sigmoid_activation)
        h0_samples = self.numpy_rng.binomial(size=h0_probability.shape , n=1 , p = h0_probability)

        pre_sigmoid_activation = np.dot(h0_samples , self.W.T) + self.vbias
        v1_probability = sigmoid(pre_sigmoid_activation)
        v1_samples = self.numpy_rng.binomial(size=v1_probability.shape , n=1 , p = v1_probability)

        pre_sigmoid_activation = np.dot(v1_samples , self.W) + self.hbias
        h1_probability = sigmoid(pre_sigmoid_activation)
        h1_samples = self.numpy_rng.binomial(size=h1_probability.shape , n=1 , p = h1_probability)
        
        return [h1_samples , v1_samples , h1_probability , v1_probability]
    
        
        
    def train(self , learning_rate = 0.05 , k = 1 , epochs = 15, batch_size=10000 , monitor=10):
        """
        Contrastive Divergence
        """
        ix0s , bsize , n_batchsize = get_minibatch(batch_size , self.input.shape[0])
        start = time.time()
       

        #train rbm by cd
        for epoch in xrange(epochs):
            dw , dv , dh = [0,0,0] #initialiazation
            idx = np.random.randint(0 , n_batchsize)*batch_size
            #for idx in xrange(n_batchsize):
            dw_var , dv_var , dh_var = self.contrastive_divergence(self.input[idx:idx+batch_size] , k=k)
            dw += dw_var
            dv += dv_var
            dh += dh_var
                #if epoch is 0:
                #set_trace()
            #dw , dv , dh = self.contrastive_divergence(self.input[idx:idx+batch_size], lr=lerning_rate , k=k)
            self.W += learning_rate * (dw / float(n_batchsize) )
            self.vbias += learning_rate * (dv / float(n_batchsize) )
            self.hbias += learning_rate * (dh / float(n_batchsize) )
            if monitor != None and epoch % monitor == 0:
                print "epoch : %d" %( (epoch))
            
        self.elapsed_time = time.time() - start
        print ("学習時間:{0}".format(self.elapsed_time)) + "[sec]"
        #return [self.W ,self.hbias , self.vbias]

    def reconstruct(self, v):
        h_probability = sigmoid(np.dot(v, self.W) + self.hbias)
        h_samples = self.numpy_rng.binomial(size=h_probability.shape , n=1 , p = h_probability)
        v_probability = sigmoid(np.dot(h_samples, self.W.T) + self.vbias)
        reconstructed_v = self.numpy_rng.binomial(size=v_probability.shape , n=1 , p = v_probability)
        return reconstructed_v
    
    def get_samples(self, v):
        h_probability = sigmoid(np.dot(v, self.W) + self.hbias)
        h_samples = self.numpy_rng.binomial(size=h_probability.shape , n=1 , p = h_probability)
        return h_samples
        
def get_minibatch(batch_size , nSamples):
    bsize = np.clip(batch_size , 0 , nSamples)
    n_batchsize = nSamples / bsize
    imax = nSamples - bsize
    ix0s = np.arange(0,imax+1 , (imax+1)/n_batchsize).astype(int)
    #ixss = [np.arange(ix0 , ix0 + bsize) for ix0 in ix0s]

    return ix0s , bsize , n_batchsize

# def free_energy(v_sample , h_sample , w , vbias , hbias):
#     bsize = v_sample.shape[0]
#     e1 = np.mean(np.sum(vbias * v_sample , axis=1))
#     e2 = np.mean(np.sum(hbias * h_sample , axis=1))
#     e3 = np.sum(np.dot(v_sample.T , h_sample) * w) / bsize
# 
#     return e1 - e2 - e3
            
        
        
if __name__ == "__main__":
    data = np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]])


    rng = np.random.RandomState(123)

    # construct RBM
    rbm = RBM(input=data, n_hidden=15, numpy_rng=rng)

    # train
    for epoch in xrange(1):
        rbm.train(batch_size=1 , epochs=50000,monitor=2000)
        cost = 3#rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
    v = np.array([[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0]])
    print v.shape
    print rbm.reconstruct(v)



        
