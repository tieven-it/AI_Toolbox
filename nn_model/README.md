### 主要模块

&emsp;&emsp;基础模块：前向传播，反向传播，梯度检测

&emsp;&emsp;参数更新：Gradient_descent方法，Momentum方法，Adam方法

&emsp;&emsp;梯度下降：mini_batch

&emsp;&emsp;正则化：L2正则化，drop_out正则化

&emsp;&emsp;初始化参数：He初始化，Xavier初始化

------

### 控制台

&emsp;&emsp;**num_unit**：隐藏层的单元数

&emsp;&emsp;例：num_unit = [50,100,200]

&emsp;&emsp;生成3个模型，每个模型隐藏层的单元数分别为50,100,200

&emsp;&emsp;例：num_unit = [50,100,200,500]

&emsp;&emsp;生成4个模型，每个模型隐藏层的单元数分别为50,100,200,500





&emsp;&emsp;**func_flag**：隐藏层激活函数

&emsp;&emsp;例：func_flag = [1,0,0]

&emsp;&emsp;生成1个模型，使用ReLu作为隐藏层激活函数

&emsp;&emsp;例：func_flag = [1,1,1]

&emsp;&emsp;生成3个模型，分别使用ReLu，Leaky_ReLu，Tanh作为隐藏层激活函数

------

### 报错情况

&emsp;&emsp;有时代价函数可能会出现如下警告：RuntimeWarning: divide by zero encountered in log，原因是输出层的中间变量z的数值太大，经过逻辑回归函数转换输出为1，在代价函数中出现log(1-1)的情况。

&emsp;&emsp;例：z = 40 → y' = sigmoid(z) = 1 →  log(y - y') = log(1 - 1) = log(0)

&emsp;&emsp;解决方法：1，减小学习率；2，关闭警告；

------

### 版本1.6

&emsp;&emsp;1,修改反向传播算法,在dz上除m,而不是在dw和db上除m,与数学公式相对应

### 版本1.5

&emsp;&emsp;1，添加mini_batch方法

&emsp;&emsp;2，添加Momentum方法

&emsp;&emsp;3，添加Adam方法

&emsp;&emsp;4，将激活函数移到外部库

### 版本1.4

&emsp;&emsp;1，添加drop_out正则化方法

### 版本1.3

&emsp;&emsp;1，使用均值归一化方法初始化数据

&emsp;&emsp;2，使用He方法初始化参数矩阵

&emsp;&emsp;3，不将sigmoid设为备选隐藏层激活函数

&emsp;&emsp;4，添加L2正则化

&emsp;&emsp;5，梯度检测不适用于ReLU函数，因为ReLU不是连续函数，在0处不可导

### 版本1.2

&emsp;&emsp;1，增加控制台的可选策略

### 版本1.1

&emsp;&emsp;1，构建神经网络基本框架

&emsp;&emsp;2，通过控制台自动化批量构建模型