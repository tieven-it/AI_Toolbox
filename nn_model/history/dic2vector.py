# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:30:43 2019

@author: Alex
"""
import numpy as np
import copy

def dictionary_to_vector(dictionary, index):
    
    vector = []
    for key in index:
        len_i = dictionary[key].shape[0]
        len_j = dictionary[key].shape[1]
        
        for i in range(len_i):
            for j in range(len_j):
                vector.append(dictionary[key][i, j]) 
    
    vector = np.array(vector)
    vector = vector.reshape(vector.shape[0],1)
    
    return vector

def vector_to_dictionary(vector,dictionary, parameters_index):
    
    dictionary_new = copy.deepcopy(dictionary)
    cnt = 0
    
    for key in parameters_index:
        len_i = dictionary_new[key].shape[0]
        len_j = dictionary_new[key].shape[1]
        
        for i in range(len_i):
            for j in range(len_j):
                dictionary_new[key][i][j] = vector[cnt][0]
                cnt += 1
    
    return dictionary_new


def gradient_checking(activation, lambd):
    
    np.random.seed(1)
    m = 50
    n = 10
    X = np.random.rand(n, m)
    Y = np.random.randint(0, 2, (1, m)).astype(np.float64)
    lim = 1e-4
    layer_dims = [n,5,5,3,1] 
    parameters = initialize_parameters(layer_dims, activation)
    
    #通过函数求的梯度向量
    caches, _ = forward_propagation(X, Y, parameters, activation, lambd)
    grad = backward_propagation(Y, parameters, caches, activation, lambd)
    
    #获得索引
    parameters_index = sorted(parameters.keys())
    gradient_index = []
    for i in range(len(parameters_index)):
        gradient_index.append(("d") + parameters_index[i])
    
    grad_vec = dictionary_to_vector(grad, gradient_index)
    
    #初始化检测的梯度向量
    #将参数从字典转为向量
    
    grad_check_vec = np.zeros((grad_vec.shape[0],1))
    lim_vec = np.zeros((grad_vec.shape[0],1))
    parameters_vec = dictionary_to_vector(parameters, parameters_index)
    
    #通过近似法求的梯度向量
    for i in range(grad_vec.shape[0]):

        lim_vec[i][0] = lim
        parameters_plus = vector_to_dictionary(parameters_vec + lim_vec, parameters, parameters_index)
        _, cost1 = forward_propagation(X, Y, parameters_plus, activation, lambd)
        
        parameters_minus = vector_to_dictionary(parameters_vec - lim_vec, parameters, parameters_index)
        _, cost2 = forward_propagation(X, Y, parameters_minus, activation, lambd)
        
        grad_check_vec[i][0] = (cost1 - cost2) / (2 * lim)
        lim_vec[i][0] = 0
        
    numerator = np.linalg.norm(grad_vec - grad_check_vec)
    denominator = np.linalg.norm(grad_vec) + np.linalg.norm(grad_check_vec)
    diff = numerator / denominator
    
    return diff