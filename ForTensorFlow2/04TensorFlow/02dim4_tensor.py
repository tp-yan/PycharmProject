import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal((4,32,32,3))
print(x)
net = layers.Conv2D(16,kernel_size=3)
print(net(x).shape) # 网络输出 (4, 30, 30, 16)