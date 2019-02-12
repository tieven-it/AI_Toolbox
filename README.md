# AI_Toolbox

### 内容简介

&emsp;&emsp;AI工具箱，放一些脱离框架实现的AI模型，通过手动造轮子加深对算法的理解。

### 现有模型

- nn_model
  - 特点1：批量化调试超参数；
  - 特点2：全向量化计算；
  - 支持的参数更新方法：Momentum方法，Adam方法；
  - 支持的梯度下降方法：mini_batch方法；
  - 支持的正则化方法：L2正则化，drop_out正则化；
  - 缺点：使用sigmoid为输出层的激活函数，部分情况下出现警告；


### 运行环境

&emsp;&emsp;python 3.6

### 参考资料

&emsp;&emsp;deeplearning.ai
