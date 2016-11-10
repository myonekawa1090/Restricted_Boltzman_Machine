#coding: utf-8
import NeuralNet as NN
import DeepNeuralNet as DNN
import DeepBeliefNet as DBN
import RestrictedBM
import loadMNIST
import sys
import cPickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import gzip
import DeepBeliefNet10_new as DBN10
from test.pickletester import protocols
from Tkconstants import HIDDEN
from cProfile import label
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import xticks, yticks
#from pyreadline.console.ironpython_console import color

fp = FontProperties(fname=r'C:\WINDOWS\Fonts\YuGothic.ttf', size=14)

#デバッグ関数
def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def showFigure(input , target , x_dim = 10 , y_dim = 5 , Opt="MNIST"):
    plt.close("all")
    fig , axes = plt.subplots(y_dim , x_dim ) #sharex = True , sharey  = True)
    
    if Opt == "MNIST":
        print "MNISTを表示します..."
        for i in range(x_dim):
            for j in range(y_dim):
                arr = input[y_dim*j + i].reshape(28,28)
                axes[j,i].imshow(arr , cmap=cm.binary)
                str = "target:{0}".format(target[y_dim*j + i])
                axes[j,i].set_xlabel(str, fontsize=20)
                axes[j,i].tick_params(labelbottom="off")
                axes[j,i].tick_params(labelleft="off")
        plt.subplots_adjust(hspace=0.3)
        plt.subplots_adjust(wspace=0.3)
    elif Opt == "WEIGHT":
        print "Weightを可視化します..."
        for i in range(x_dim):
            for j in range(y_dim):
                arr = input[y_dim*j + i].reshape(28,28)
                axes[j,i].imshow(arr , cmap=cm.binary)
                axes[j,i].tick_params(labelbottom="off")
                axes[j,i].tick_params(labelleft="off")
        plt.subplots_adjust(wspace=0 , hspace=0)

    
    
    plt.show()
    return True

def load_mnist_dataset(dataset):
    """
    MNISTのデータセットをダウンロードします
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

def ConvertVector2Matrix(x , size):
    # return d converted in (size , n)
    d = np.zeros(size)#np.zeros((size,10))
    for i in range(size[0]):
        d[ i , x[i] ]  = 1
    temp = 3
    return d

def augument_labels(labels, order):
    """
    1次元のラベルデータを、ラベルの種類数(order)次元に拡張します
    """
    new_labels = []
    for i in range(labels.shape[0]):
        v = np.zeros(order)
        v[labels[i]] = 1
        new_labels.append(v)
    
    return np.array(new_labels).reshape((labels.shape[0], order))        

if __name__ == "__main__":
    train_set, valid_set, test_set = load_mnist_dataset("mnist.pkl.gz")
    f = gzip.open("binary_mnist.pkl.gz" , "r")
    train_data = cPickle.load(f)
    f.close()
    train_target = ConvertVector2Matrix(train_set[1], (len(train_set[0]) , 10))    
    argv = sys.argv
#--------------------------------------------------------2-NN--------------------------------------------------------#
    if argv[1] == "mnist":
        print "MNISTを表示します..."
        showFigure(train_data, train_set[1], x_dim=5, y_dim=5, Opt="MNIST")
    if argv[1] == "shallow":
        filename = "2-layer_params.pkl"
        input_layer , hidden_layer , output_layer = (784 , 100 , 10)
        nn = NN.NeuralNet(input_layer , hidden_layer , output_layer)
        print "階層型 NN 入力層:%d, 隠れ層:%d, 出力層:%d" % (input_layer, hidden_layer, output_layer)
        print "学習を行いますか? Y/n"  #setting parameters
        usrInput = raw_input()
        if usrInput == "Y":
            epochs = 50000
            learning_rate = 0.1
            print "学習回数 :%d, 学習率:%f" % (epochs , learning_rate)
            nn.train(train_data , train_target , epochs=epochs , learning_rate=learning_rate , monitor=2000)
            params = {"W1":nn.W1 , "W2":nn.W2 , "out_bias":nn.out_bias , "h1_bias":nn.h1_bias}
            print "パラメータを保存しています..."
            f = open(filename , "w")
            cPickle.dump(params,f)
            f.close
        elif usrInput == "n":
            print "パラメータを読み込んでいます..."
            f = open(filename)
            params = cPickle.load(f)
            nn.W1 = params["W1"]
            nn.W2 = params["W2"]
            nn.out_bias = params["out_bias"]
            nn.h1_bias = params["h1_bias"]
            f.close()
        else:
            print "コマンドが間違っています"
            sys.exit()
            
        print "テストを行いますか? Y/n" #using parameters
        usrInput = raw_input()
        if usrInput == "Y":
            #Test Phase
            # テスト
            test_data, test_target = test_set
            results    = np.arange(len(test_data), dtype=np.int)
            for n in range(len(test_data)):
                results[n] = nn.predict(test_data[n])
                # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
            print "正解率: ", (results == test_target).mean()
#             actual_data = nn.forward(test_set[0])
#             output_class = np.argmax(actual_data , axis=1)
            showFigure(test_data[:50] , results[:50])
        elif usrInput == "n":
            print "プログラムを終了します."
        elif usrInput =="debug":
            #For Debug
            data = input.data[0:3]
            target = input.target[0:3]
            #set_trace()
            actual_data = nn.forward(data)
            output_class = np.argmax(actual_data,axis=1)
            print actual_data , output_class , target
            
        else:
            print "コマンドが間違っています."
#-------------------------------------------------------binary化--------------------------------------------------------#
    elif argv[1] == "binary":
        print "mnistを２値化しています..."
        j_size = train_data[:,0].size
        i_size = train_data[0].size
        binary_data = train_data[:]
        for j in range(j_size):
            for i in range(i_size):
                if train_data[j , i] >= 0.5:
                    binary_data[j , i] = 1
                else:   
                    binary_data[j,i] = 0        
        print "binary_mnist.pklに保存しています..."
        f = gzip.open("binary_mnist.pkl.gz" , "wb")
        cPickle.dump(binary_data , f , protocol=2)
        f.close()
#--------------------------------------------------------rbm--------------------------------------------------------#
    elif argv[1] == "rbm":
        n_hidden = 100
        print "RBM 隠れ層:%d" %(n_hidden)
        learning_rate , k , epochs , batch_size = (0.1 , 1 , 1000 , 100)
        print "学習回数 :%d , 学習率:%f , k:%d , バッチサイズ:%d" % (epochs , learning_rate,  k , batch_size)
        filename = "1rbm_params.pkl"
        #print "binary_mnist_pkl.gz を読み込んでいます..."
        rbm = RestrictedBM.RBM(input=train_data , n_hidden=n_hidden)
        print "学習を行いますか? Y/n"
        usrInput = raw_input()
        if usrInput == "Y":
            rbm.train(learning_rate=learning_rate, k=k, epochs=epochs, batch_size=batch_size , monitor=2000)
            params = {"W":rbm.W , "vbias":rbm.vbias , "hbias":rbm.hbias , "elapsed_time":rbm.elapsed_time}
            print "パラメータを保存しています..."
            f = open(filename , "w")
            cPickle.dump(params,f)
            f.close
        elif usrInput == "n":
            print "パラメータを読み込んでいます..."
            f = open(filename)
            params = cPickle.load(f)
            rbm.W = params["W"]
            rbm.vbias = params["vbias"]
            rbm.hbias = params["hbias"]
            rbm.elapsed_time = params["elapsed_time"]
            f.close()
        print "重みを可視化しますか? Y/n"
        usrInput = raw_input()
        if usrInput == "Y":
            showFigure(rbm.W.T, None, x_dim=10, y_dim=10, Opt="WEIGHT")
        
#--------------------------------------------------------5-NN--------------------------------------------------------#
    elif argv[1] == "deep":
        filename = "5-layer_params.pkl"
        input_layer , hidden1 , hidden2 , hidden3 , output_layer = (784 , 100 , 100 , 100 , 10)
        dnn = DNN.DeepNeuralNet(input_layer, hidden1, hidden2 , hidden3 , output_layer)
        strTmp = "階層型NN:事前学習なし, 入力層:%d, 隠れ層1:%d, 隠れ層2:%d, 隠れ層3:%d, 出力層:%d" % (input_layer, hidden1, hidden2, hidden3, output_layer)
        print strTmp
        epochs = 10000
        learning_rate = 0.1
        strTmp = "学習回数 :%d, 学習率:%f" % (epochs , learning_rate) 
        print strTmp
        print "学習を行いますか? Y/n"  #setting parameters
        usrInput = raw_input()
        if usrInput == "Y":
            dnn.train(train_data , train_target , epochs=epochs , learning_rate=learning_rate , monitor=100)
            params = {"Message":strTmp ,"W1":dnn.W1 , "W2":dnn.W2 ,"W3":dnn.W3 ,"W4":dnn.W4 ,
                      "out_bias":dnn.out_bias , "h1_bias":dnn.h1_bias , "h2_bias":dnn.h2_bias , "h3_bias":dnn.h3_bias ,
                       "elapsed_time":dnn.elapsed_time , "cost":dnn.cost}
            print "パラメータを保存しています..."
            f = open(filename , "w")
            cPickle.dump(params,f)
            f.close
        elif usrInput == "n":
            print "パラメータを読み込んでいます..."
            f = open(filename)
            params = cPickle.load(f)
            dnn.W1 = params["W1"]
            dnn.W2 = params["W2"]
            dnn.W3 = params["W3"]
            dnn.W4 = params["W4"]
            dnn.h1_bias = params["h1_bias"]
            dnn.h2_bias = params["h2_bias"]
            dnn.h3_bias = params["h3_bias"]
            dnn.out_bias = params["out_bias"]
            dnn.elapsed_time = params["elapsed_time"]
            f.close()
        else:
            print "コマンドが間違っています"
            sys.exit()
            
        print "テストを行いますか? Y/n" #using parameters
        usrInput = raw_input()
        if usrInput == "Y":
            #Test Phase
            # テスト
            print "前回の学習時間:{0}".format(dnn.elapsed_time) + "[sec]"
            test_data, test_target = test_set
            results    = np.arange(len(test_data), dtype=np.int)
            for n in range(len(test_data)):
                results[n] = dnn.predict(test_data[n])
                # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
            print "正解率: ", (results == test_target).mean()
#             actual_data = nn.forward(test_set[0])
#             output_class = np.argmax(actual_data , axis=1)
            showFigure(test_data[:50] , results[:50])
    elif argv[1] == "belief":
        filename = "5-layer_dbn_params.pkl"
        rbm1name = "rbm1_params.pkl"
        rbm2name = "rbm2_params.pkl"
        rbm3name = "rbm3_params.pkl"
        rbm4name = "rbm4_params.pkl"
        input_layer , hidden1 , hidden2 , hidden3 , output_layer = (784 , 100 , 100 , 100 , 10)  
        strTmp = "階層型NN:事前学習あり, 入力層:%d, 隠れ層1:%d, 隠れ層2:%d, 隠れ層3:%d, 出力層:%d" % (input_layer, hidden1, hidden2, hidden3, output_layer)
        print strTmp
        epochs = 10000
        learning_rate = 0.1
        strTmp = "学習回数 :%d, 学習率:%f" % (epochs , learning_rate)
        print strTmp
        print "学習を行いますか? Y/n"  #setting parameters
        usrInput = raw_input()
        if usrInput == "Y":
            dbn = DBN.DeepBeliefNet(train_data , True)
            dbn.train(train_data , train_target , epochs=epochs , learning_rate=learning_rate , monitor=100)
            params = {"Message":strTmp , "W1":dbn.W1 , "W2":dbn.W2 ,"W3":dbn.W3 ,"W4":dbn.W4 ,
                      "out_bias":dbn.out_bias , "h1_bias":dbn.h1_bias , "h2_bias":dbn.h2_bias , "h3_bias":dbn.h3_bias , 
                      "elapsed_time":dbn.elapsed_time , "cost":dbn.cost}
            print "パラメータを保存しています..."
            f = open(filename , "w")
            cPickle.dump(params,f)
            f.close
        elif usrInput == "n":
            print "パラメータを読み込んでいます..."
            dbn = DBN.DeepBeliefNet(train_data , False)
            f = open(filename)
            params = cPickle.load(f)
            dbn.W1 = params["W1"]
            dbn.W2 = params["W2"]
            dbn.W3 = params["W3"]
            dbn.W4 = params["W4"]
            dbn.h1_bias = params["h1_bias"]
            dbn.h2_bias = params["h2_bias"]
            dbn.h3_bias = params["h3_bias"]
            dbn.out_bias = params["out_bias"]
            dbn.elapsed_time = params["elapsed_time"]
            f.close()
        else:
            print "コマンドが間違っています"
            sys.exit()
            
        print "テストを行いますか? Y/n" #using parameters
        usrInput = raw_input()
        if usrInput == "Y":
            #Test Phase
            # テスト
            print "前回の学習時間:{0}".format(dbn.elapsed_time) + "[sec]"
            test_data, test_target = test_set
            results    = np.arange(len(test_data), dtype=np.int)
            for n in range(len(test_data)):
                results[n] = dbn.predict(test_data[n])
                # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
            print "正解率: ", (results == test_target).mean()
#             actual_data = nn.forward(test_set[0])
#             output_class = np.argmax(actual_data , axis=1)
            showFigure(test_data[:50] , results[:50])   
    elif argv[1] == "cost":
        
        print "階層型NNと, 事前学習を用いた階層型NNの誤差関数をプロットします..."
        str1 = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\"+ argv[2] + "_dnn.pkl"
        str2 = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\" + argv[2] + "_dbn.pkl"
        f = open(str1)
        params = cPickle.load(f)
        Message = params["Message"]
        dnn_cost = params["cost"]
        dnn_acc = params["accuracy"]
        dnn_time = params["elapsed_time"]
        f.close()
        f = open(str2)
        params = cPickle.load(f)
        dbn_cost = params["cost"]
        dbn_acc = params["accuracy"]
        dbn_time = params["elapsed_time"]
        f.close()
        f = open("C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\mean.txt" , "a+") 
        f.write(Message)
        f.write(" : ")
        str = "DNN 識別率: {0} , 学習時間 : {1}[sec]".format(dnn_acc , dnn_time)
        f.write(str)
        str = "DBN 識別率 : {0} , 学習時間 : {1}[sec]\n".format(dbn_acc , dbn_time)
        f.write(str)
        f.close()
        plt.close("all")
        h_axis = []
        temp = len(dnn_cost[0])
        for i in range(temp):
            h_axis.append(i*100)
            
        plt.plot(h_axis,dbn_cost[0] , label="pre training ON" , color="b") 
        for i in range(9):
            plt.plot(h_axis , dbn_cost[i+1] , color="b")
        
        plt.plot(h_axis , dnn_cost[0] , label="pre training OFF" , color="g")
        for i in range(9):
            plt.plot(h_axis , dnn_cost[i+1] , color="g")
            
        plt.xlabel(u"学習回数    ", fontsize = 40 , fontproperties=fp)
        plt.ylabel(u"誤差関数" , fontsize = 40 , fontproperties = fp)
        xticks(fontsize=40)
        yticks(fontsize=40)
        plt.legend(fontsize=40)
        plt.show()
    elif argv[1] == "mean":
        epochs = int(argv[3])
        temp = epochs / 100
        learning_rate = float(argv[4])
        strTmp = "学習回数 :%d, 学習率:%f" % (epochs , learning_rate)
        print strTmp
#         accuracy = 0
#         costArr = []
#         elapsed_time = 0
#         str1 = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\" + argv[2] + "_dnn.pkl"
#         for i in range(10):
#             print "DNN : %d" % (i)
#             dnn = DNN.DeepNeuralNet(784, 100, 100, 100, 10)
#             dnn.train(train_data , train_target , epochs=epochs , learning_rate=learning_rate , monitor=100)
#             test_data, test_target = test_set
#             results    = np.arange(len(test_data), dtype=np.int)
#             for n in range(len(test_data)):
#                 results[n] = dnn.predict(test_data[n])
#                 # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
#             accuracy += (results == test_target).mean()
#             elapsed_time += dnn.elapsed_time
#             costArr.append(dnn.cost) 
#                 
#         accuracy /= 10
#         elapsed_time /= 10
#         params = {"Message":strTmp , "accuracy":accuracy , "elapsed_time":elapsed_time , "cost":costArr}
#         print "パラメータを保存しています..."
#         f = open(str1 , "w")
#         cPickle.dump(params,f)
#         f.close

        accuracy = 0
        costArr = []
        elapsed_time = 0
        str2 = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\" + argv[2] + "_dbn.pkl"
        for i in range(10):
            print "DBN : %d" % (i)
            dbn = DBN.DeepBeliefNet(train_data, False)
            dbn.train(train_data , train_target , epochs=epochs , learning_rate=learning_rate , monitor=100)
            test_data, test_target = test_set
            results    = np.arange(len(test_data), dtype=np.int)
            for n in range(len(test_data)):
                results[n] = dbn.predict(test_data[n])
                # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
            accuracy += (results == test_target).mean()
            elapsed_time += dbn.elapsed_time
            costArr.append(dbn.cost) 
        accuracy /= 10
        elapsed_time /= 10
        params = {"Message":strTmp , "accuracy":accuracy , "elapsed_time":elapsed_time , "cost":costArr}
        print "パラメータを保存しています..."
        f = open(str2, "w")
        cPickle.dump(params,f)
        f.close
        
    elif argv[1] == "10layer":
        if argv[3] == "True":
            filename = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\10layer_dbn.pkl"
            PreTrainFlag = True
        else:
            filename = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\10layer_dnn.pkl"
            PreTrainFlag = False
        if argv[2] =="True":
            LearnFlag=True
        else:
            LearnFlag=False;
        input_layer , hidden1 , hidden2 , hidden3 , hidden4 , hidden5 , hidden6 , hidden7 , hidden8 , output_layer = (784 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 50 , 10)  
        epochs = 10000
        learning_rate = 0.1
        strTmp = "学習回数 :%d, 学習率:%f" % (epochs , learning_rate)
        print strTmp
        print "学習を行いますか? Y/n"  #setting parameters
        usrInput = raw_input()
        if usrInput == "Y":
            dbn = DBN10.DeepBeliefNet10(train_data , LearnFlag , PreTrainFlag)
            dbn.train(train_data , train_target , epochs=epochs , learning_rate=learning_rate , monitor=100)
            params = {"Message":strTmp , "W1":dbn.W1 , "W2":dbn.W2 ,"W3":dbn.W3 ,"W4":dbn.W4 ,"W5":dbn.W5 , "W6":dbn.W6 , "W7":dbn.W7 ,"W8":dbn.W8 ,"W9":dbn.W9 ,
                      "out_bias":dbn.out_bias , "h1_bias":dbn.h1_bias , "h2_bias":dbn.h2_bias , "h3_bias":dbn.h3_bias , "h4_bias":dbn.h4_bias , "h5_bias":dbn.h5_bias , "h6_bias":dbn.h6_bias , 
                      "h7_bias":dbn.h7_bias ,  "h8_bias":dbn.h8_bias ,
                      "elapsed_time":dbn.elapsed_time , "cost":dbn.cost}
            print "パラメータを保存しています..."
            f = open(filename , "w")
            cPickle.dump(params,f)
            f.close
        elif usrInput == "n":
            print "パラメータを読み込んでいます..."
            dbn = DBN.DeepBeliefNet(train_data , False)
            f = open(filename)
            params = cPickle.load(f)
            dbn.W1 = params["W1"]
            dbn.W2 = params["W2"]
            dbn.W3 = params["W3"]
            dbn.W4 = params["W4"]
            dbn.W5 = params["W5"]
            dbn.W6 = params["W6"]
            dbn.W7 = params["W7"]
            dbn.W8 = params["W8"]
            dbn.W9 = params["W9"]
            
            dbn.h1_bias = params["h1_bias"]
            dbn.h2_bias = params["h2_bias"]
            dbn.h3_bias = params["h3_bias"]
            dbn.h4_bias = params["h4_bias"]
            dbn.h5_bias = params["h5_bias"]
            dbn.h6_bias = params["h6_bias"]
            dbn.h7_bias = params["h7_bias"]
            dbn.h8_bias = params["h8_bias"]
            
            dbn.out_bias = params["out_bias"]
            dbn.elapsed_time = params["elapsed_time"]
            f.close()
        else:
            print "コマンドが間違っています"
            sys.exit()
            
        print "テストを行いますか? Y/n" #using parameters
        usrInput = raw_input()
        if usrInput == "Y":
            #Test Phase
            # テスト
            print "前回の学習時間:{0}".format(dbn.elapsed_time) + "[sec]"
            test_data, test_target = test_set
            results    = np.arange(len(test_data), dtype=np.int)
            for n in range(len(test_data)):
                results[n] = dbn.predict(test_data[n])
                # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
            print "正解率: ", (results == test_target).mean()
#             actual_data = nn.forward(test_set[0])
#             output_class = np.argmax(actual_data , axis=1)
            showFigure(test_data[:50] , results[:50])   
        
        
    elif argv[1] == "cost2":
        
        print "階層型NNと, 事前学習を用いた階層型NNの誤差関数をプロットします..."
        str1 = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\"+ argv[2] + "_dnn.pkl"
        str2 = "C:\\Users\\Masaki\\GoogleDrive\\Labo\\data\\" + argv[2] + "_dbn.pkl"
        f = open(str1)
        params = cPickle.load(f)
        Message = params["Message"]
        dnn_cost = params["cost"]
        dnn_time = params["elapsed_time"]
        f.close()
        f = open(str2)
        params = cPickle.load(f)
        dbn_cost = params["cost"]
        dbn_time = params["elapsed_time"]
        f.close()
        plt.close("all")
        h_axis = []
        temp = len(dnn_cost)
        for i in range(temp):
            h_axis.append(i*100)
            
        plt.plot(h_axis,dbn_cost , label="pre training ON" , color="b") 
        plt.plot(h_axis , dnn_cost , label="pre training OFF" , color="g")
        plt.xlabel(u"学習回数    ", fontsize = 10 , fontproperties=fp)
        plt.ylabel(u"誤差関数" , fontsize = 10 , fontproperties = fp)
        xticks(fontsize=5)
        yticks(fontsize=5)
        plt.legend(fontsize=10)
        plt.show()    
    else :
        print "Command is Not Correct"
    print "プログラムを終了します"
