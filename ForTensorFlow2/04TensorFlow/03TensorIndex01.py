import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.python基本索引
a = tf.ones([1, 5, 5, 3])
print(a[0][0])
print(a[0][0][0])
print(a[0][0][0][2])

# 2.Numpy-style indexing
a = tf.random.normal([4, 28, 28, 3])
print(a[1].shape)
print(a[1, 2].shape)  # a[1,2] ==> a[1][2]
print(a[1, 2, 3].shape)
print(a[1, 2, 3, 2].shape)
# 2.1 切片 start:end
a =tf.range(10)
print(a)
print(a[-1:])
print(a[:2])
print(a[:-1])
# 2.2 Indexing by ':'
a = tf.random.normal([4, 28, 28, 3])
print(a.shape)
print(a[0].shape)
print(a[0,:,:,:].shape)
print(a[0,1,:,:].shape)
print(a[:,:,:,0].shape)
print(a[:,:,:,2].shape)
print(a[:,0,:,2].shape)
# 2.3 Indexing by '::'
# start:end:step
print(a.shape)
print(a[0:2,:,:,:].shape)
print(a[:,0:28:2,0:28:2,:].shape)
print(a[:,:14,:14,:].shape)
print(a[:,15:,15:,:].shape)
print(a[:,::2,::2,:].shape)
# 逆序采样，::-1
a=tf.range(4)
print(a[::-1])
print(a[::-2])
print(a[2::-2])
# 2.4 ‘...’若干个‘:’的缩写，到底几个由python根据上下文自己确定
a = tf.random.normal([2,4, 28, 28, 3])
print(a[0].shape)
print(a[0,:,:,:,:].shape)
print(a[0,...].shape) # ... ==> :,:,:,:
print(a[:,:,:,:,0].shape)
print(a[...,0].shape)
print(a[0,...,2].shape)
print(a[1,0,...,0].shape) # ... ==> :,:

