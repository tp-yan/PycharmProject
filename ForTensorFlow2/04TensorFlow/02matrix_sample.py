import tensorflow as tf
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal([4,784])
print(x.shape)

net = layers.Dense(10)
net.build((4,784)) # 假设输入层为 [4,784]，构建模型
print(net.kernel.shape,net.kernel)
print(net(x).shape) #
print(net.kernel.shape,net.kernel)
print(net.bias.shape)