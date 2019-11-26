import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1.输出二分类0,1
# `tf.sigmoid`：将实数集压缩到0~1
a = tf.linspace(-6.,6,10)
print(tf.sigmoid(a)) # 不保证和为1

x = tf.random.normal([1,28,28])*5
print(tf.reduce_min(x),tf.reduce_max(x))
x = tf.sigmoid(x)
print(tf.reduce_min(x),tf.reduce_max(x))

# 2.tf.nn.softmax：需要输出每种类别的概率，且保证概率和为1。
a = tf.linspace(-2.,2,10)
print(tf.sigmoid(a))
print(tf.nn.softmax(a)) # softmax一般作用于logits上，将强者变得更强，扩大了值间倍数

# 3.tf.tanh：输出-1~1。sigmoid的偏移版
print(tf.tanh(a))
