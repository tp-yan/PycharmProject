import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a=np.arange(5)
print(a.dtype)
aa = tf.convert_to_tensor(a) # numpy --> tensor.
print(aa) # 默认int64
aa = tf.convert_to_tensor(a,dtype=tf.int32)
print(aa)

tf.cast(aa,dtype=tf.float32) # cast:专用于类型转化
aaa = tf.cast(aa,dtype=tf.float64)
print(aaa)
print(tf.cast(aaa,dtype=tf.int32))

# bool <--> int
b = tf.constant([0,1])
bb = tf.cast(b,dtype=tf.bool)
print(bb)
print(tf.cast(bb,dtype=tf.int32))