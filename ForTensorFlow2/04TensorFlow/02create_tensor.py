import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF中支持：int、float、double、bool、string
# TF中Tensor(类)是一切数据的载体，而数据自身有其属性类型。
# tf.constant与tf.convert_to_tensor一样。
print(tf.constant(1))
print(tf.constant([1]))
print(tf.constant([1],2.))
# print(tf.constant([1,2.],[3.])) # 必须保证list内元素维度一致
print(tf.constant(1.))
# print(tf.constant(1.,dtype=tf.int32))
print(tf.constant(2.2,dtype=tf.double)) # 指定数据类型必须与数据类型一致. tf.double == float64

print(tf.constant([True,False]))
print(tf.constant("hello,world."))


print("=============== create tensor ===============")
# 1. create tensor From Numpy, List
# 1.1 From Numpy
a = tf.convert_to_tensor(np.ones([2,3]))
print(a)
print(tf.convert_to_tensor(np.zeros([2,3])))
# 1.2 From List：要保证 List可以转换成Numpy array，否则不能转换成Tensor
b = tf.convert_to_tensor([1,2])
print(b)
print(tf.convert_to_tensor([1,2.])) # 自动将 list的int元素转为float
print(tf.convert_to_tensor([[1],[2.]])) # 需要list 每个元素的维度一致

# 2. from tf 自带的方法
# 2.1 tf.zeros(shape)
print(tf.zeros([])) # 标量
print(tf.zeros([1])) # 向量
print(tf.zeros([2,3]))
print(tf.zeros([2,3,4]))
# 2.2 tf.zeros_like(tensor)
a = tf.zeros([2,3,4])
print(tf.zeros_like(a))
print(tf.zeros(a.shape))
# 2.3 tf.ones(shape)
print(tf.ones(1)) # 一维向量，长度为1. ==> tf.ones([1])
print(tf.ones([1]))
print(tf.ones(2)) # ==> tf.ones([2])
print(tf.ones([2]))
print(tf.ones([])) # 标量
print(tf.ones([2,3]))
# 2.4 tf.ones_like(shape)
a = tf.ones([2,3])
print(tf.ones_like(a))
# 2.5 tf.fill(shape,value)
print(tf.fill([2,2],0))
print(tf.fill([2,2],1.))
print(tf.fill([2,2],True))
# 2.6 正态分布随机数
a = tf.random.normal([2,2],mean=1,stddev=1)
print(a)
print(tf.random.normal([2,2]))
print(tf.random.truncated_normal([2,2],mean=0,stddev=1))
# 2.7 均匀分布
print(tf.random.uniform([2,2],minval=0,maxval=10,dtype=tf.int32)) # 指定随机数类型为int
print(tf.random.uniform([2,2],minval=0,maxval=100))

# 3.Random Permutation
def random_permutation():
    """
    随机置乱样本。置乱样本索引即可，让样本和标签使用同一的索引即可保证样本与标签对应。
    :return:
    """
    idx = tf.range(10) # 假设索引范围0~9
    idx = tf.random.shuffle(idx) # 打乱索引
    print("----------------------------------------------")
    print(idx)
    print("----------------------------------------------")
    a = tf.random.normal([10,784]) # 假设有10个样本
    b = tf.random.uniform([10],maxval=10,dtype=tf.int32) # 样本对应标签
    print(a)
    print(b)

    a = tf.gather(a,idx) # 按索引重新置乱
    b = tf.gather(b,idx)
    print(a,b)
random_permutation()

