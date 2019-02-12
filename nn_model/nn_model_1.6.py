# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:30:54 2019
1,修改反向传播算法,在dz上除m,而不是在dw和db上除m,与数学公式相对应

@author: Alex
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
import scipy.io
import sklearn
import sklearn.datasets
from models.activation import relu,leaky_relu,sigmoid,tanh,relu_gradient,\
                              leaky_relu_gradient,sigmoid_gradient,tanh_gradient

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 导入数据
def load_dataset():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])

    # 尺寸转换
    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    train_x = train_x_orig.reshape(m_train, -1).T
    test_x = test_x_orig.reshape(m_test, -1).T
    train_y = train_y_orig.reshape(1, train_y_orig.shape[0])
    test_y = test_y_orig.reshape(1, test_y_orig.shape[0])

    return train_x, train_y, test_x, test_y


def load_2D_dataset():

    data = scipy.io.loadmat('datasets/data_moon.mat')
    train_x = data['X'].T
    train_y = data['y'].T
    test_x = data['Xval'].T
    test_y = data['yval'].T

    return train_x, train_y, test_x, test_y

def load_2D_moon_dataset():
    
    np.random.seed(3)
    train_x, train_y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
    train_x = train_x.T
    train_y = train_y.reshape(1, train_y.shape[0])
    np.random.seed(2)
    test_x, test_y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
    test_x = test_x.T
    test_y = test_y.reshape(1, test_y.shape[0])
    
    return train_x, train_y, test_x, test_y


# 初始化参数
def initialize_parameters(layer_dims):

    np.random.seed(1)        # 随机数种子
    parameters = {}
    L = len(layer_dims) - 1  # 总层数

    for i in range(1, L+1):
        # he方法初始化（2/...）
        # xavier 方法初始化（1/...）
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


# 前向传播
def forward_propagation(X, Y, parameters, activation, lambd, keep_prob):

    # np.random.seed(1)
    L = len(parameters) // 2
    caches = {}
    temp_cost = 0
    m = X.shape[1]
    caches["A0"] = np.copy(X)  # 深复制

    # 第1~L-1层
    for i in range(1, L):
        
        # 核心部分
        caches["Z"+str(i)] = np.dot(parameters["W"+str(i)], caches["A"+str(i-1)]) + parameters["b"+str(i)]
        caches["A"+str(i)] = globals().get(activation[0])(caches["Z"+str(i)])
       
        # drop_out正则化
        caches["D"+str(i)] = np.random.rand(caches["A"+str(i)].shape[0], caches["A"+str(i)].shape[1]) < keep_prob
        caches["A"+str(i)] = caches["A"+str(i)] * caches["D"+str(i)]
        caches["A"+str(i)] = caches["A"+str(i)] / keep_prob
        
        # L2正则化部分的代价函数
        temp_cost += np.sum(np.square(parameters["W"+str(i)]))

    # 第L层
    caches["Z"+str(L)] = np.dot(parameters["W"+str(L)], caches["A"+str(L-1)]) + parameters["b"+str(L)]
    caches["A"+str(L)] = globals().get(activation[1])(caches["Z"+str(L)])
    temp_cost += np.sum(np.square(parameters["W"+str(L)]))

    # 代价函数
    cost_part1 = (-1./m) * np.sum(Y * np.log(caches["A"+str(L)]) + (1-Y) * np.log(1-caches["A"+str(L)]))
    cost_part2 = lambd * temp_cost / (2 * m)
    cost = cost_part1 + cost_part2

    return caches, cost


# 反向传播
def backward_propagation(Y, parameters, caches, activation, lambd, keep_prob):

    L = len(parameters) // 2
    grads = {}
    m = Y.shape[1]

    # 第L层
    caches["dZ"+str(L)] = (1./m) * (caches["A"+str(L)] - Y)
    grads["dW"+str(L)] = np.dot(caches["dZ"+str(L)], caches["A"+str(L-1)].T) +\
        lambd / m * parameters["W"+str(L)]
    grads["db"+str(L)] = np.sum(caches["dZ"+str(L)], axis=1, keepdims=True)

    # 第L-1层
    for i in reversed(range(1, L)):
        
        # drop_out正则化
        caches["dA"+str(i)] = np.dot(parameters["W"+str(i+1)].T, caches["dZ"+str(i+1)])
        caches["dA"+str(i)] = caches["dA"+str(i)] * caches["D"+str(i)] / keep_prob
       
        # 核心部分
        temp_gradient = globals().get(activation[0]+"_gradient")(caches["Z"+str(i)])
        caches["dZ"+str(i)] = caches["dA"+str(i)] * temp_gradient
        grads["dW"+str(i)] = np.dot(caches["dZ"+str(i)], caches["A"+str(i-1)].T) +\
            lambd / m * parameters["W"+str(i)]
        grads["db"+str(i)] = np.sum(caches["dZ"+str(i)], axis=1, keepdims=True)

    return grads


# 梯度检测
def gradient_checking(activation, lambd, keep_prob):

    """
    说明：当激活函数为relu或leaky_relu时，梯度检测的值没有参考价值，梯度检测应用的前提是
         函数为连续且处处可导，relu或leaky_relu不是连续函数，且在x=0处不可导。

    原因：以relu函数为例，理论上当x=0时没有导数，而在编程中通常是将x=0处的导数设为0或者1。
         即使x的值不为0，当x的绝对值小于误差时，导数也可能不准确。

    举例：假设x=1e-5，误差e=1e-4，理论上该点导数应该等于1，实际却等于0.55。
         f(x+e) = f(0.00011) = 0.00011, f(x-e) = f(-0.00009) = 0
         f'(x) = (f(x+e)-f(x-e)) / (2*e) = 0.55
    """
    np.random.seed(1)
    m = 5                       # 样本数
    n = 10                      # 特征数
    X = np.random.randn(n, m)   # 随机初始化m个样本，每个样本n个特征
    Y = np.random.randint(0, 2, (1, m))                         # 随机初始化m个标签
    lim = 1e-4                                                  # 误差值
    layer_dims = [n, 5, 5, 3, 1]                                # 神经网络规格
    parameters = initialize_parameters(layer_dims)              # 随机初始化参数
    grad_init_vec = []                                          # 累加元素的平方
    grad_std_vec = []

    # 通过函数求导数
    caches, cost_init = forward_propagation(X, Y, parameters, activation, lambd, keep_prob)
    grad_init = backward_propagation(Y, parameters, caches, activation, lambd, keep_prob)
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
                caches, cost1 = forward_propagation(X, Y, parameters, activation, lambd, keep_prob)
                parameters[key][i, j] = temp - lim
                caches, cost2 = forward_propagation(X, Y, parameters, activation, lambd, keep_prob)

                # 计算梯度
                grad_std["d"+key][i, j] = (cost1 - cost2) / (2 * lim)
                parameters[key][i, j] = temp
                grad_init_vec.append(grad_init["d"+key][i, j])
                grad_std_vec.append(grad_std["d"+key][i, j])

    grad_init_vec = np.array(grad_init_vec)
    grad_std_vec = np.array(grad_std_vec)
    diff = np.linalg.norm(grad_init_vec - grad_std_vec) / (np.linalg.norm(grad_init_vec) + np.linalg.norm(grad_std_vec))

    return diff


# 生成mini_batch
def random_mini_batch(X, Y, batch_size, seed):

    np.random.seed(seed)
    m = X.shape[1]
    mini_batchs = []
    if batch_size == 0:                        # 使用batch方法
        batch_size = m
        
    assert(0 < batch_size <= m)                # 检测数值范围
    assert(isinstance(batch_size, int))        # 检测数值类型

    # 打乱排序
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # 确定完整的块数
    num_batchs = np.floor(m / batch_size).astype(int)
    for i in range(num_batchs):
        mini_batch_X = shuffled_X[:, i * batch_size: (i+1) * batch_size]
        mini_batch_Y = shuffled_Y[:, i * batch_size: (i+1) * batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batchs.append(mini_batch)

    # 如果最后一块不完整
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, num_batchs * batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_batchs * batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batchs.append(mini_batch)

    return mini_batchs


# 初始化梯度方法对应的参数
def initialize_optimizer(parameters, optimizer):
    
    L = len(parameters) // 2
    caches = {}
    assert( 0 < optimizer < 4)
    assert(isinstance(optimizer, int))
    
    # gradient方法
    if optimizer == 1:    
        pass
    
    # momentum方法
    elif optimizer == 2:   
        for i in range(1, L+1):
            caches["Vdw"+str(i)] = np.zeros(parameters["W"+str(i)].shape)
            caches["Vdb"+str(i)] = np.zeros(parameters["b"+str(i)].shape)
    
    # adam方法
    elif optimizer == 3:
        for i in range(1, L+1):
            caches["Vdw"+str(i)] = np.zeros(parameters["W"+str(i)].shape)
            caches["Vdb"+str(i)] = np.zeros(parameters["b"+str(i)].shape)
            caches["Sdw"+str(i)] = np.zeros(parameters["W"+str(i)].shape)
            caches["Sdb"+str(i)] = np.zeros(parameters["b"+str(i)].shape)
            
    return caches


# 更新参数
def update_parameters(parameters, alpha, grads, caches, optimizer, cnt):
    
    L = len(parameters) // 2
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    correct = {}
    
    # gradient方法
    if optimizer == 1:
        for i in range(1, L+1):
            parameters["W"+str(i)] = parameters["W"+str(i)] - alpha * grads["dW"+str(i)]
            parameters["b"+str(i)] = parameters["b"+str(i)] - alpha * grads["db"+str(i)]
    
    # momentum方法
    elif optimizer == 2:
        for i in range(1, L+1):
            caches["Vdw"+str(i)] = beta1 * caches["Vdw"+str(i)] + (1-beta1) * grads["dW"+str(i)]
            caches["Vdb"+str(i)] = beta1 * caches["Vdb"+str(i)] + (1-beta1) * grads["db"+str(i)]
            parameters["W"+str(i)] = parameters["W"+str(i)] - alpha * caches["Vdw"+str(i)]
            parameters["b"+str(i)] = parameters["b"+str(i)] - alpha * caches["Vdb"+str(i)]
    
    # adam方法
    elif optimizer == 3:
        for i in range(1, L+1):
            
            # 累加梯度和梯度的平方
            caches["Vdw"+str(i)] = beta1 * caches["Vdw"+str(i)] + (1-beta1) * grads["dW"+str(i)]
            caches["Vdb"+str(i)] = beta1 * caches["Vdb"+str(i)] + (1-beta1) * grads["db"+str(i)]
            caches["Sdw"+str(i)] = beta2 * caches["Sdw"+str(i)] + (1-beta2) * np.square(grads["dW"+str(i)])
            caches["Sdb"+str(i)] = beta2 * caches["Sdb"+str(i)] + (1-beta2) * np.square(grads["db"+str(i)])
            
            # 偏差修正
            correct["Vdw"+str(i)] = caches["Vdw"+str(i)] / (1 - np.power(beta1, cnt))
            correct["Vdb"+str(i)] = caches["Vdb"+str(i)] / (1 - np.power(beta1, cnt))
            correct["Sdw"+str(i)] = caches["Sdw"+str(i)] / (1 - np.power(beta2, cnt))
            correct["Sdb"+str(i)] = caches["Sdb"+str(i)] / (1 - np.power(beta2, cnt))
            
            # 梯度更新
            parameters["W"+str(i)] = parameters["W"+str(i)] - alpha * correct["Vdw"+str(i)] \
                                     / (np.sqrt(correct["Sdw"+str(i)]) + epsilon)
            parameters["b"+str(i)] = parameters["b"+str(i)] - alpha * correct["Vdb"+str(i)] \
                                     / (np.sqrt(correct["Sdb"+str(i)]) + epsilon)
            
    
    return  parameters, caches


# 训练模型
def training(X, Y, parameters, alpha, maxIters, activation, lambd, keep_prob, batch_size, optimizer):

    costs = []
    seed = 0
    cnt = 0
    caches_gradient = initialize_optimizer(parameters, optimizer)
    
    for j in range(maxIters):
        seed = seed + 1
        mini_batchs = random_mini_batch(X, Y, batch_size, seed)
        cost = 0

        for i in mini_batchs:
            cnt = cnt + 1
            mini_batch_X = i[0]
            mini_batch_Y = i[1]
            caches_propagation, cost = forward_propagation(mini_batch_X, mini_batch_Y, parameters,     # 前向传播
                                                           activation, lambd, keep_prob)
            grads = backward_propagation(mini_batch_Y, parameters, caches_propagation, activation,     # 反向传播
                                         lambd, keep_prob)
            parameters, caches_gradient = update_parameters(parameters, alpha, grads, caches_gradient, # 参数更新 
                                                            optimizer, cnt)                             

        costs.append(cost)
        
        # 学习率衰减
        # if j % 50 == 0:
            # alpha = alpha * 0.5
        
        if j % 10 == 0:
            print("\r" + "percent of training : %d" % (j / maxIters * 100) + "%", end="")

    return parameters, costs


# 预测准确率
def predict(X, Y, parameters, activation, lambd=0, keep_prob=1):

    L = len(parameters) // 2
    caches, cost = forward_propagation(X, Y, parameters, activation, lambd, keep_prob)
    Y_predict = caches["A"+str(L)] > 0.5
    prediction = (np.dot(Y_predict, Y.T) + np.dot(1-Y_predict, 1-Y.T)) / float(Y.shape[1]) * 100

    return np.squeeze(prediction)


# 画边界图
def plot_decision_boundary(X, Y, parameters, activation, lambd=0, keep_prob=1):

    L = len(parameters) // 2
    x_min, x_max = X[0, :].min() - 2, X[1, :].max() + 2
    y_min, y_max = X[0, :].min() - 2, X[1, :].max() + 2
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    test_x = np.c_[xx.ravel(), yy.ravel()].T
    test_y = np.random.randint(0, 2, (1, test_x.shape[1]))
    caches, _ = forward_propagation(test_x, test_y, parameters, activation, lambd, keep_prob)
    Z = caches["A"+str(L)] > 0.5
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plt.show()

    return


# 模型主体
def nn_model(alpha, maxIters, layer_dims, activation, lambd, keep_prob, batch_size, optimizer):

    # 导入数据
    train_x, train_y, test_x, test_y = load_dataset()

    # 特征归一化
    mu = np.mean(train_x, axis=1).reshape(train_x.shape[0], 1)
    sigma = np.std(train_x, axis=1).reshape(train_x.shape[0], 1)
    train_x = (train_x - mu) / sigma
    test_x = (test_x - mu) / sigma

    # 初始化参数
    parameters = initialize_parameters(layer_dims)

    # 梯度检测
    diff = gradient_checking(activation, lambd, keep_prob)
    print("the error of gradient is %.2e" % diff)

    # 训练参数
    parameters, costs = training(train_x, train_y, parameters, alpha, maxIters,
                                 activation, lambd, keep_prob, batch_size, optimizer)

    # 画代价函数
    plt.plot(costs)
    plt.show()

    # 预测准确性
    prediction_train = predict(train_x, train_y, parameters, activation)
    prediction_test = predict(test_x, test_y, parameters, activation)

    # 画边界图
    # plot_decision_boundary(train_x, train_y, parameters, activation)

    # 评分
    score = 2 * (prediction_train/100 * prediction_test/100) / (prediction_train/100 + prediction_test/100)

    return prediction_train, prediction_test, score, costs


# 调用模型
def call_model(record, alpha, maxIters, layer_dims, activation, lambd, keep_prob, batch_size, optimizer):

    record["alpha"] = alpha
    record["layer_dims"] = layer_dims
    record["activation"] = activation
    record["lambd"] = lambd
    prediction_train, prediction_test, score, costs = nn_model(alpha, maxIters, layer_dims, activation, 
                                                               lambd, keep_prob, batch_size, optimizer)
    print("\n")

    record["prediction_train"] = prediction_train
    record["prediction_test"] = prediction_test
    record["score"] = score
    record["costs"] = costs

    return record


# 策略一：一个隐藏层，不同的单元数
def strategy_1(history, func, alpha, maxIters, func_flag, num_unit, num_features,
                lambd, keep_prob, batch_size, optimizer):
    cnt = 0
    num_choice = len(num_unit) * func_flag.count(1)

    for i in range(len(num_unit)):
        for j in range(len(func_flag)):

            layer_dims = [num_features, num_unit[i], 1]
            activation = [func[j], "sigmoid"]

            if func_flag[j] == 1:
                print("%d / %d" % (cnt+1, num_choice))
                history["test"+str(cnt)] = {}
                history["test"+str(cnt)] = call_model(history["test"+str(cnt)], alpha, maxIters,
                                                      layer_dims, activation, lambd, keep_prob, 
                                                      batch_size, optimizer)
                cnt = cnt + 1

    return history


# 策略二：n个隐藏层，相同的单元数
def strategy_2(history, func, alpha, maxIters, func_flag, num_unit, num_layer, 
                num_features, lambd, keep_prob, batch_size, optimizer):

    cnt = 0
    num_choice = len(num_layer) * func_flag.count(1)

    for i in range(len(num_layer)):
        layer_dims = [num_features]

        for j in range(num_layer[i]):
            layer_dims.append(num_unit)

        layer_dims.append(1)

        for k in range(len(func_flag)):
            activation = [func[k], "sigmoid"]

            if func_flag[k] == 1:
                print("%d / %d" % (cnt+1, num_choice))
                history["test"+str(cnt)] = {}
                history["test"+str(cnt)] = call_model(history["test"+str(cnt)], alpha, maxIters,
                                                      layer_dims, activation, lambd, keep_prob, 
                                                      batch_size, optimizer)
                cnt = cnt+1

    return history


# 控制台
def control_strategy(label):
    
    # 默认设置：学习率每迭代100减半,关
    # 默认设置：beta1 = 0.9
    # 默认设置：beta2 = 0.999
    
    func = ["relu", "leaky_relu", "tanh"]
    num_features = 12288
    history = {}
    alpha = 0.01
    maxIters = 100
    lambd = 0                     # 0:关闭L2
    keep_prob = 1                 # 1:关闭drop_out
    batch_size = 0                # 0:关闭mini_batch
    optimizer = 1                 # 1:gradient; 2:momentum; 3:adam

    # 策略一：一个隐藏层，不同的单元数
    if label == 1:
        num_unit = [50, 100, 200]
        func_flag = [1, 0, 0]
        history = strategy_1(history, func, alpha, maxIters, func_flag, num_unit,
                             num_features, lambd, keep_prob, batch_size, optimizer)

    # 策略二：n个隐藏层，相同的单元数/逻辑回归模型
    if label == 2:
        num_unit = 100                  # 确定最佳单元数
        func_flag = [1, 0, 0]           # 选择激活函数
        num_layer = [1,2,3]             # 确定隐藏层层数（当num_layer为0时，是逻辑回归模型）
        history = strategy_2(history, func, alpha, maxIters, func_flag, num_unit,
                             num_layer, num_features, lambd, keep_prob, batch_size, optimizer)

    # 策略三：自定义
    if label == 3:
        activation = ["relu", "sigmoid"]
        layer_dims = [num_features,100,1]
        history["test0"] = {}
        history["test0"] = call_model(history["test0"], alpha, maxIters, layer_dims,
                                      activation, lambd, keep_prob, batch_size, optimizer)

    for i in range(len(history)):
        print("\n", history["test"+str(i)]["activation"]+history["test"+str(i)]["layer_dims"])
        print("prediction_test:" , history["test"+str(i)]["prediction_test"])
        print("prediction_train:" , history["test"+str(i)]["prediction_train"])
        print("score:", history["test"+str(i)]["score"])

    return history

control_strategy(1)