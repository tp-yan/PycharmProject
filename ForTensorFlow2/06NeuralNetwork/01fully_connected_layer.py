import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.keras并不是之前的keras，而是TF对keras的实现，是TF的一部分。

# 1.构建单层全连接网络
def single_layer():
    # Dense：全连接层。一层网络包含了权重weights、偏置bias、激活函数、节点输出值
    net = tf.keras.layers.Dense(10)
    # print(net.bias) # 网络还未构建，所以还没有创建 weights和bias
    print(net.get_weights)
    print(net.weights)
    net.build(input_shape=(None, 4))  # 构建网络需要确定输入维度。构建网络并随机初始化weights和bias置零
    print(net.kernel.shape)  # net.kernel就是weights的Tensor格式
    print(net.weights)
    print(net.bias.shape)
    print(net.bias)

    net.build(input_shape=(None, 20))  # 输入不同维度可以构建不同的网络
    print(net.kernel.shape)
    print(net.bias.shape)

    net.build(input_shape=(2, 4))
    print(net.kernel)

def net_forward():
    x = tf.random.normal([4,784])
    net = tf.keras.layers.Dense(512)
    out = net(x) # 向网络输入特征向量。若此时还未构建网络，则按照特征向量维度调用build构建,然后进行前向传播，返回网络输出
    print(out.shape)
    print(net.bias.shape)
    print(net.kernel.shape)

def multi_layer():
    # 用层容器keras.Sequential将多个单层放在一起，每个元素是个Dense
    x = tf.random.normal([2,3])
    model = keras.Sequential([
        layers.Dense(2,activation='relu'),
        layers.Dense(2,activation='relu'),
        layers.Dense(2)
    ])
    model.build(input_shape=(None,3))
    model.summary() # model.summary --> print：打印网络结构

    for p in model.trainable_variables: # model.trainable_variables：返回所有可训练参数变量
        print(p.name,p.shape)

single_layer()
net_forward()
multi_layer()