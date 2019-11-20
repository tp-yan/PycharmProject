import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF中支持：int、float、double、bool、string
# TF中Tensor(类)是一切数据的载体，而数据自身有其属性类型。
print(tf.constant(1))
print(tf.constant(1.))
# print(tf.constant(1.,dtype=tf.int32))
print(tf.constant(2.2,dtype=tf.double)) # 指定数据类型必须与数据类型一致. tf.double == float64

print(tf.constant([True,False]))
print(tf.constant("hello,world."))



