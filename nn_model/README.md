### 主要模块

&emsp;&emsp;基础模块：前向传播，反向传播，梯度检测

&emsp;&emsp;参数更新：Gradient_descent方法，Momentum方法，Adam方法

&emsp;&emsp;梯度下降：mini_batch

&emsp;&emsp;正则化：L2正则化，drop_out正则化

&emsp;&emsp;初始化参数：He初始化，Xavier初始化

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