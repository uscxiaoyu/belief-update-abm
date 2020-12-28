## 一、更新历史

### 1. 更新2020.12.28
- 重新定义了一些变量的名称，使其更能表达内在含义
- 分离agent属性：按照属性是否在模拟过程中动态变化，在`__init__()`和`init_state()`两个函数中分别设置，其中`init_state()`中设置的参数在每次模拟`diffuse()`中重新初始化
- 设置了agent对创新为好创新的置信度`innovGood_degree`：随机取一个整数值作为其置信度，然后利用`logit`函数将其转换为概率；在处理消息的过程中，如果处理了一个正面口碑，则`innovGood_degree := innovGood_degree + 1`；如果处理了一个负面口碑，则`innovGood_degree := innovGood_degree - 1`。
- 实现绘图
- 实现模拟层面的多进程`multi_diffuse_cores`


## 二、论文内容
### 1. 扩散算法


### 2. 种子节点策略