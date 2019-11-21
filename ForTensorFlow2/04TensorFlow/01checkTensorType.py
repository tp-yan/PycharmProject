import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 检查Tensor类型用 tf.is_tensor()
a = tf.constant([1.])
b = tf.constant([True, False])
c = tf.constant("hello")
d = np.arange(4)  # numpy array

print(isinstance(a, tf.Tensor))
print(tf.is_tensor(b))
print(tf.is_tensor(d))
print(a.dtype, b.dtype, c.dtype)
print(a.dtype == tf.float32)
print(c.dtype == tf.string)
