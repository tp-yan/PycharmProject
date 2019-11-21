import tensorflow as tf
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

net = layers.Dense(10)
net.build((4,8)) # 假设输入层为 [4,8]，构建模型
print(net.kernel.shape,net.kernel) # 权重参数w一般随机初始化
print(net.bias) # 偏置一般初始化为0

