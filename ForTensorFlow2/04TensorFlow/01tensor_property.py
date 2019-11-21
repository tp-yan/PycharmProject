import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tensor本质是携带数据的Tensor实例，拥有方法和属性
# Tensor处于指定设备，只能在该设备上进行运算，不能混用
with tf.device("cpu"):
    a = tf.constant([1])
with tf.device("gpu"):
    b = tf.range(4)
    print(b.device,type(b),b.numpy())

# 转换设备，创建新Tensor
aa = a.gpu()
bb = b.cpu()

print(b.numpy()) # 转为Numpy类型，只能由CPU处理
print(b.ndim) # Tensor维度
print(tf.rank(b)) # 同 .ndim，但是返回Tensor
print(tf.rank(tf.ones([3,4,2])))
# print(b.name) # 这个属性没卵用，TF1.0遗留
