#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle
from sklearn import datasets

#デバッグ関数
def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def loadInputData():
    f = open("binary_mnist.pkl" , "r")
    data_input = pickle.load(f)
    f.close()

    return data_input

def readInputData():
    
    print "mnistを読み込んでいます..."
    mnist = datasets.fetch_mldata("MNIST original" , "~/")
    data_input = mnist
    
    print "binary_mnist.pklに保存しています..."
    f = open("binary_mnist.pkl" , "w+")
    pickle.dump(data_input , f)
    f.close()
                
    return data_input
