import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.concat
# 需要指定在哪个维度axis上合并，concat不会改变维度，只是同维度下的累加 。
# 限制：合并维度之外的维度大小必须相同
a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
c = tf.concat([a, b], axis=0)  # 在第一维上累叠
print(c)

a = tf.ones([4, 35, 8])
b = tf.ones([4, 3, 8])
print(tf.concat([a, b], axis=1).shape)

# 2.stack
# 会增加一个新维度。
# 限制：所有维度大小相同
a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
print(tf.concat([a, b], axis=-1).shape)
c = tf.stack([a, b], axis=0)  # axis指定增加维度的插入位置
print(c)
print(tf.stack([a, b], axis=3).shape)
print(tf.stack([a, b], axis=2).shape)

# 3.unstack
# 只能均匀分割，分割长度固定为1，split的一种特殊情况。
c = tf.stack([a,b])
print(c.shape)
aa,bb = tf.unstack(c,axis=0) # 指定axis有多长，则分割出多少个变量
print(aa.shape,bb.shape)
res = tf.unstack(c,axis=3)
print(res[0].shape,res[7].shape)

# 4.split
# 可任意指定分割长度和份数
print(len(tf.unstack(c,axis=3))) # 分割长度固定为1
res=tf.split(c,axis=3,num_or_size_splits=2) # 每份分割长度为2
print(len(res))
print(res[0].shape)
res = tf.split(c,axis=3,num_or_size_splits=[2,2,4]) # 指定每份长度
print(res[0].shape,res[2].shape) # (2, 4, 35, 2) (2, 4, 35, 4)