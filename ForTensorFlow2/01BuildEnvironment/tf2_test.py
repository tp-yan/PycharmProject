import tensorflow as tf
import os

print(os.environ) # Ubuntu的环境变量，包括PATH、SHELL、PWD等环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 减少Tensorflow打印出的无关信息

a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)

print("GPU:", tf.test.is_gpu_available())
