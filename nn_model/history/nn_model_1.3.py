# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:30:54 2019
1,使用均值归一化方法初始化数据
2,使用He方法初始化参数矩阵
3,不将sigmoid设为备选隐藏层激活函数
3,添加L2正则化
4,梯度检测不适用于ReLU函数，因为ReLU不是连续函数，在0处不可导

@author: Alex
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy

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
    train_y = train_y_orig.reshape(1, train_y_orig.shape[0])
    test_y = test_y_orig.reshape(1, test_y_orig.shape[0])   
    
    # 转换数据类型(若不转换为浮点型，np.log会出现警告)
    train_y = train_y.astype(np.float64)
    test_y = test_y.astype(np.float64)

    return train_x, train_y, test_x, test_y

# 初始化参数
def initialize_parameters(layer_dims, activation):
      
    np.random.seed(1)        # 随机数种子
    parameters = {}
    L = len(layer_dims) - 1  # 总层数
    
    for i in range(1,L+1):
        # 用he方法初始化
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
        
    return parameters

# 前向传播
def forward_propagation(X, Y, parameters, activation, lambd):
    
    L = len(parameters) // 2 
    caches = {}
    temp_cost = 0
    m = X.shape[1]   
    caches["A0"] = np.copy(X)  # 深复制
    
    # 第1~L-1层
    for i in range(1, L):
        caches["Z"+str(i)] = np.dot(parameters["W"+str(i)], caches["A"+str(i-1)]) + parameters["b"+str(i)]
        caches["A"+str(i)] = globals().get(activation[0])(caches["Z"+str(i)])
        temp_cost += np.sum(np.square(parameters["W"+str(i)]))
    # 第L层
    caches["Z"+str(L)] = np.dot(parameters["W"+str(L)], caches["A"+str(L-1)]) + parameters["b"+str(L)] 
    caches["A"+str(L)] = globals().get(activation[1])(caches["Z"+str(L)]) 
    temp_cost += np.sum(np.square(parameters["W"+str(L)]))
    
    cost_part1 = (-1/m) * np.sum(Y * np.log(caches["A"+str(L)]) + (1-Y) * np.log(1-caches["A"+str(L)]))
    cost_part2 = lambd / (2 * m) * temp_cost
    cost = cost_part1 + cost_part2
    
    return caches, cost

# 反向传播
def backward_propagation(Y, parameters, caches, activation, lambd):
    
    L = len(parameters) // 2
    grads = {}
    m = Y.shape[1]
    
    # 第L层
    caches["dZ"+str(L)] = caches["A"+str(L)] - Y
    grads["dW"+str(L)] = (1/m) * np.dot(caches["dZ"+str(L)], caches["A"+str(L-1)].T) + lambd / m * parameters["W"+str(L)]
    grads["db"+str(L)] = (1/m) * np.sum(caches["dZ"+str(L)], axis=1, keepdims=True)
    
    # 第L-1层
    for i in reversed(range(1, L)):
        temp_gradient = globals().get(activation[0]+"_gradient")(caches["Z"+str(i)])
        caches["dZ"+str(i)] = np.dot(parameters["W"+str(i+1)].T, caches["dZ"+str(i+1)]) * temp_gradient
        grads["dW"+str(i)] = (1/m) * np.dot(caches["dZ"+str(i)], caches["A"+str(i-1)].T) + lambd / m * parameters["W"+str(i)]
        grads["db"+str(i)] = (1/m) * np.sum(caches["dZ"+str(i)], axis=1, keepdims=True)
    
    return grads    

# 梯度检测
def gradient_checking(activation, lambd):
    
    '''
    说明：当激活函数为relu或leaky_relu时，梯度检测的值没有参考价值，梯度检测应用的前提是
         函数为连续且处处可导，relu或leaky_relu不是连续函数，且在x=0处不可导。     
         
    原因：以relu函数为例，理论上当x=0时没有导数，而在编程中通常是将x=0处的导数设为0或者1。
         即使x的值不为0，当x的绝对值小于误差时，导数也可能不准确。   
         
    举例：假设x=1e-5，误差e=1e-4，理论上该点导数应该等于1，实际却等于0.55。
         f(x+e) = f(0.00011) = 0.00011, f(x-e) = f(-0.00009) = 0
         f'(x) = (f(x+e)-f(x-e)) / (2*e) = 0.55    
    '''
    np.random.seed(1)
    m = 5                       # 样本数
    n = 10                      # 特征数
    X = np.random.randn(n, m)   # 随机初始化m个样本，每个样本n个特征
    Y = np.random.randint(0, 2, (1, m)).astype(np.float64)      # 随机初始化m个标签
    lim = 1e-4                                                  # 误差值
    layer_dims = [n,5,5,3,1]                                    # 神经网络规格
    parameters = initialize_parameters(layer_dims, activation)  # 随机初始化参数
    grad_init_vec = []                                          # 累加元素的平方   
    grad_std_vec = []
    
    # 通过函数求导数
    caches, cost_init = forward_propagation(X, Y, parameters, activation, lambd)
    grad_init = backward_propagation(Y, parameters, caches, activation, lambd)
    grad_std = copy.deepcopy(grad_init)                         # 深复制字典
    
    # 遍历字典中数组
    for key, item in parameters.items():
        len_i, len_j = item.shape     
        
        # 遍历数组中元素
        for i in range(len_i):          
            for j in range(len_j):  
                   
                # 计算代价
                temp = item[i, j]
                parameters[key][i, j] = temp + lim                
                caches, cost1 = forward_propagation(X, Y, parameters, activation, lambd)                
                parameters[key][i, j] = temp - lim
                caches, cost2 = forward_propagation(X, Y, parameters, activation, lambd)
                
                # 计算梯度
                grad_std["d"+key][i, j] = (cost1 - cost2) / (2 * lim)
                parameters[key][i, j] = temp
                grad_init_vec.append(grad_init["d"+key][i, j])
                grad_std_vec.append(grad_std["d"+key][i, j])
    
    grad_init_vec = np.array(grad_init_vec)
    grad_std_vec = np.array(grad_std_vec)
    diff = np.linalg.norm(grad_init_vec - grad_std_vec) / (np.linalg.norm(grad_init_vec) + np.linalg.norm(grad_std_vec))
    
    return diff

# 训练模型
def training(X, Y, parameters, alpha, maxIters, layer_dims, activation, lambd):
    
    costs = []
    L = len(layer_dims) - 1
    
    # 梯度下降
    for j in range(maxIters):               
        caches, cost = forward_propagation(X, Y, parameters, activation, lambd)   # 前向传播       
        grads = backward_propagation(Y, parameters, caches, activation, lambd)    # 反向传播
        costs.append(cost)   
        
        for i in range(1,L+1):                                                    # 梯度更新    
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

# 神经网络模型
def nn_model(alpha, maxIters, layer_dims, activation, lambd):
    
    # 导入数据
    train_x, train_y, test_x, test_y = load_dataset()   
    
    # 特征归一化
    mu = np.mean(train_x, axis=1).reshape(train_x.shape[0],1)
    sigma = np.std(train_x, axis=1).reshape(train_x.shape[0],1)
    train_x = (train_x - mu) / sigma
    test_x = (test_x - mu) / sigma
    
    # 初始化参数
    parameters = initialize_parameters(layer_dims, activation)    
    
    # 梯度检测
    test = gradient_checking(activation, lambd)
    print("the error of gradient is %.2e" % test)
    
    # 训练参数
    parameters, costs = training(train_x, train_y, parameters, alpha, maxIters, layer_dims, activation, lambd)
    plt.plot(costs)
    
    # 预测准确性
    prediction_train = predict(train_x, train_y, parameters, activation)
    prediction_test = predict(test_x, test_y, parameters, activation)
    
    # 评分
    score = 2 * (prediction_train/100 * prediction_test/100) / (prediction_train/100 + prediction_test/100)
    
    return prediction_train, prediction_test, score, costs

# 记录策略
def call_model(record, alpha, maxIters, layer_dims, activation, lambd):
          
    record["alpha"] = alpha
    record["layer_dims"] = layer_dims
    record["activation"] = activation
    record["lambd"] = lambd
    prediction_train, prediction_test, score, costs = nn_model(alpha, maxIters, layer_dims, activation, lambd)
    print("\n")
    
    record["prediction_train"] = prediction_train
    record["prediction_test"] = prediction_test
    record["score"] = score
    record["costs"] = costs
    
    return record

# 阶段一：一个隐藏层，不同的单元数    
def strategy_1(history, func, alpha, maxIters, func_flag, num_unit, num_features, lambd):
    
    cnt = 0;
    num_choice = len(num_unit) * func_flag.count(1)
    
    for i in range(len(num_unit)):
        for j in range(len(func_flag)):
            
            layer_dims = [num_features, num_unit[i], 1]
            activation = [func[j], "sigmoid"]
            
            if func_flag[j] == 1:
                print("%d / %d" %(cnt+1, num_choice))
                history["test"+str(cnt)] = {}
                history["test"+str(cnt)] = call_model(history["test"+str(cnt)], alpha, maxIters, layer_dims, activation, lambd)            
                cnt = cnt + 1
                
    return history

# 阶段二：n个隐藏层，相同的单元数/逻辑回归模型
def strategy_2(history, func, alpha, maxIters, func_flag, num_unit, num_layer, num_features, lambd):
    
    cnt = 0;   
    num_choice = len(num_layer) * func_flag.count(1)
    
    for i in range(len(num_layer)):
        layer_dims = [num_features]
        
        for j in range(num_layer[i]):
            layer_dims.append(num_unit)
        
        layer_dims.append(1)
        
        for k in range(len(func_flag)):
            activation = [func[k], "sigmoid"]
            
            if func_flag[k] == 1:
                print("%d / %d" %(cnt+1, num_choice))
                history["test"+str(cnt)] = {}
                history["test"+str(cnt)] = call_model(history["test"+str(cnt)], alpha, maxIters, layer_dims, activation, lambd)            
                cnt = cnt+1

    return history

# 控制策略
def control_strategy(label):
    
    func = ["relu", "leaky_relu", "tanh"]
    num_features = 12288
    history = {}
    alpha = 0.01
    lambd = 1
    maxIters = 200
    
    #阶段一：一个隐藏层，不同的单元数
    if label == 1:
        num_unit = [100]
        func_flag = [1,1,1]
        history = strategy_1(history, func, alpha, maxIters, func_flag, num_unit, num_features, lambd)
        
    # 阶段二：n个隐藏层，相同的单元数/逻辑回归模型
    if label == 2: 
        num_unit = 100               # 确定最佳单元数
        func_flag = [1,1,0]          # 选择激活函数
        num_layer = [5]              # 确定隐藏层层数（当num_layer为0时，是逻辑回归模型）
        history = strategy_2(history, func, alpha, maxIters, func_flag, num_unit, num_layer, num_features, lambd)
    
    for i in range(len(history)):      
        print("\n",history["test"+str(i)]["activation"]+history["test"+str(i)]["layer_dims"],"\n")
        print(history["test"+str(i)]["score"],"\n")
     
    return history

history = control_strategy(2)