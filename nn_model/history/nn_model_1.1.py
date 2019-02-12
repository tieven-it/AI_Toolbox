# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:30:54 2019
1,构建神经网络基本框架
2,通过控制台自动化批量构建模型
@author: Alex
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
import seaborn as sns


# 随机数种子
np.random.seed(1)

# 激活函数
def relu(Z):

    A = np.maximum(0, Z)
    
    return A

def leaky_relu(Z):
    
    A = np.ones((Z.shape))
    A[Z < 0] = 0.01
    A  = A * Z
    
    return A
    
def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    
    return A

def tanh(Z):
    
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + (np.exp(-Z)))
    
    return A

# 激活函数的导数
def relu_gradient(Z):
    
    A = np.ones((Z.shape))
    A[Z < 0] = 0
    
    return A

def leaky_relu_gradient(Z):
    
    A = np.ones((Z.shape))
    A[Z < 0] = 0.01
    
    return A

def sigmoid_gradient(Z):
    
    A = sigmoid(Z) * (1 - sigmoid(Z))
    
    return A

def tanh_gradient(Z):
    
    A = 1 - np.power(tanh(Z),2)
    
    return A

# 导入数据
def load_dataset():
    
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])
    
    # 尺寸转换
    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    train_x = train_x_orig.reshape(m_train,-1).T
    test_x = test_x_orig.reshape(m_test,-1).T
    
    # 转换数据类型(若不转换为浮点型，np.log会出现警告)
    train_y = train_y_orig.reshape(1, train_y_orig.shape[0])
    test_y = test_y_orig.reshape(1, test_y_orig.shape[0]) 
    
    train_y = train_y.astype(np.float64)
    test_y = test_y.astype(np.float64)

    
    return train_x, train_y, test_x, test_y

# 初始化参数
def initialize_parameters(layer_dims):
      
    np.random.seed(1)        # 随机数种子
    parameters = {}
    L = len(layer_dims) - 1  # 总层数
    
    for i in range(1,L+1):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
        
    return parameters

# 前向传播
def forward_propagation(X, Y, parameters, activation):
    
    L = len(parameters) // 2 
    caches = {}
    m = X.shape[1]
    
    caches["A0"] = X
    
    # 第1~L-1层
    for i in range(1, L):
        caches["Z"+str(i)] = np.dot(parameters["W"+str(i)], caches["A"+str(i-1)]) + parameters["b"+str(i)]
        caches["A"+str(i)] = globals().get(activation[0])(caches["Z"+str(i)])
    
    # 第L层
    caches["Z"+str(L)] = np.dot(parameters["W"+str(L)], caches["A"+str(L-1)]) + parameters["b"+str(L)] 
    caches["A"+str(L)] = globals().get(activation[1])(caches["Z"+str(L)])
    
    cost = (-1/m) * np.sum(Y * np.log(caches["A"+str(L)]) + (1-Y) * np.log(1-caches["A"+str(L)]), axis=1)
    cost = np.squeeze(cost)
    
    return caches, cost

# 反向传播
def backward_propagation(Y, parameters, caches, activation):
    
    L = len(parameters) // 2
    grads = {}
    m = Y.shape[1]
    
    # 第L层
    grads["dZ"+str(L)] = caches["A"+str(L)] - Y
    grads["dW"+str(L)] = (1/m) * np.dot(grads["dZ"+str(L)], caches["A"+str(L-1)].T)
    grads["db"+str(L)] = (1/m) * np.sum(grads["dZ"+str(L)], axis=1, keepdims=True)
    
    # 第L-1
    for i in reversed(range(1, L)):
        temp_gradient = globals().get(activation[0]+"_gradient")(caches["Z"+str(i)])
        grads["dZ"+str(i)] = np.dot(parameters["W"+str(i+1)].T, grads["dZ"+str(i+1)]) * temp_gradient
        grads["dW"+str(i)] = (1/m) * np.dot(grads["dZ"+str(i)], caches["A"+str(i-1)].T)
        grads["db"+str(i)] = (1/m) * np.sum(grads["dZ"+str(i)], axis=1, keepdims=True)
    
    return grads    

# 代价和梯度
def cost_and_gradient(X, Y, parameters, activation):
    
    caches, cost = forward_propagation(X, Y, parameters, activation)
    grads = backward_propagation(Y, parameters, caches, activation)
    
    return cost, grads

# 梯度检测
def gradient_checking(activation):
    
    np.random.seed(1)
    m = 3                     # 样本数
    n = 5                     # 特征数
    X = np.random.randn(n, m) # 随机初始化m个样本，每个样本n个特征
    Y = np.random.randint(0, 2, (1, m))            # 随机初始化m个标签
    lim = 1e-4                                     # 误差值
    layer_dims = [n,2, 1]                          # 神经网络规格
    parameters = initialize_parameters(layer_dims) # 随机初始化参数
    norm2_init = 0                                 # 累加元素的平方   
    norm2_std = 0
    
    # 通过函数求导数
    cost_init, grad_init = cost_and_gradient(X, Y, parameters, activation)
    
    # 深复制字典,清空元素的值
    grad_std = copy.deepcopy(grad_init)
    
    # 遍历字典中数组
    for key, item in parameters.items():
        # 遍历数组中元素
        len_i, len_j = item.shape
        
        for i in range(len_i):
            for j in range(len_j):
                # 计算代价
                temp = item[i, j]
                parameters[key][i, j] = temp + lim
                cost1, grad_temp = cost_and_gradient(X, Y, parameters, activation)
                parameters[key][i, j] = temp - lim
                cost2, grad_temp = cost_and_gradient(X, Y, parameters, activation)
                # 计算梯度
                grad_std["d"+key][i, j] = (cost1 - cost2) / (2 * lim)
                parameters[key][i, j] = temp
                norm2_init += np.power(grad_init["d"+key][i, j], 2)
                norm2_std += np.power(grad_std["d"+key][i, j],2)
    
    err = (norm2_init - norm2_std) / (norm2_init + norm2_std)
    
    return err

# 神经网络模型
def nn_model(X, Y, alpha, maxIters, layer_dims, activation):
    
    costs = []
    L = len(layer_dims) - 1
    parameters = initialize_parameters(layer_dims)    # 初始化参数
    
    # 梯度下降
    for j in range(maxIters):
        
        cost, grads = cost_and_gradient(X, Y, parameters, activation)
        costs.append(cost)
        
        # 梯度更新
        for i in range(1,L+1):
            
            parameters["W"+str(i)] = parameters["W"+str(i)] - alpha * grads["dW"+str(i)]
            parameters["b"+str(i)] = parameters["b"+str(i)] - alpha * grads["db"+str(i)]
        
        if j % 10 == 0:
            print("\r"+"percent of training : %d" % (j / maxIters * 100)+"%" ,end="")
    
    return parameters, costs

# 预测准确率    
def predict(X, Y, parameters, activation):
    
    L = len(parameters) // 2 
    caches = {}
    caches["A0"] = X
     
    # 第1~L-1层
    for i in range(1, L):
        caches["Z"+str(i)] = np.dot(parameters["W"+str(i)], caches["A"+str(i-1)]) + parameters["b"+str(i)]
        caches["A"+str(i)] = globals().get(activation[0])(caches["Z"+str(i)])
    
    # 第L层
    caches["Z"+str(L)] = np.dot(parameters["W"+str(L)], caches["A"+str(L-1)]) + parameters["b"+str(L)] 
    caches["A"+str(L)] = globals().get(activation[1])(caches["Z"+str(L)])        
    
    Y_predict = caches["A"+str(L)] > 0.5
    prediction = (np.dot(Y_predict, Y.T) + np.dot(1-Y_predict, 1-Y.T)) / float(Y.shape[1]) * 100
    
    return np.squeeze(prediction)


# 可视化参数
def plot(parameters):
    
    n = len(parameters) // 2 - 1
    
    for i in range(n):
        plt.subplot(1,n,i+1)
        data = parameters["W"+str(i+1)]
        sns.heatmap(data, cmap="Reds")
        # sns.heatmap(data, cmap="Reds", mask=((data > 1e-5)|(data < -1e-5)))
    plt.show()
    
    return
  
def main(alpha, maxIters, layer_dims, activation):
    # 导入数据
    train_x, train_y, test_x, test_y = load_dataset()
    
    # 特征归一化
    mu = np.mean(train_x)
    sigma = np.std(train_x)
    train_x = (train_x - mu) / sigma
    test_x = (test_x - mu) / sigma
    
    # 梯度检测
    err = gradient_checking(activation)
    print("the error of gradient is %.2e\n" % err)
    
    # 深度神经网络
    parameters, costs = nn_model(train_x, train_y, alpha, maxIters, layer_dims, activation)
    
    # 画图
    plt.plot(costs)
    # plot(parameters)
    
    # 预测准确性
    prediction_train = predict(train_x, train_y, parameters, activation)
    prediction_test = predict(test_x, test_y, parameters, activation)
    
    # 评分
    score = 2 * (prediction_train/100 * prediction_test/100) / (prediction_train/100 + prediction_test/100)
    
    return prediction_train, prediction_test, score, costs

def strategy(record, alpha, maxIters, layer_dims, activation):
          
    record["alpha"] = alpha
    record["layer_dims"] = layer_dims
    record["activation"] = activation
    
    prediction_train, prediction_test, score, costs = main(alpha, maxIters, layer_dims, activation)
    
    record["prediction_train"] = prediction_train
    record["prediction_test"] = prediction_test
    record["score"] = score
    record["costs"] = costs
    
    return record
    

def control_strategy():
    
    func = ["sigmoid", "tanh", "relu", "leaky_relu"]

    cnt = 1
    history = {}
    maxIters = 200
    
    # 阶段一：逻辑回归
    alpha = 0.01
    layer_dims = [12288,1]
    activation = [func[0], func[0]]
    history["test0"] = strategy({}, alpha, maxIters, layer_dims, activation)
    
    # 阶段二：一个隐藏层
    alpha = 0.1
    num = [100]
    func_flag = [0,0,1,1]
    
    for i in range(len(num)): 
        
        for j in range(len(func_flag)):
            
            layer_dims = [12288, num[i], 1]
            activation = [func[j], func[0]] 
            
            if func_flag[j] == 1:
                history["test"+str(cnt)] = strategy({}, alpha, maxIters, layer_dims, activation)
                cnt = cnt + 1
   
    for i in range(len(history)):      
   
        print("\n",history["test"+str(i)]["activation"]+history["test"+str(i)]["layer_dims"],"\n")
        print(history["test"+str(i)]["score"],"\n")
     
    return history

history = control_strategy()