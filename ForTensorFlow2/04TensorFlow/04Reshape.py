import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensor属性： shape, ndim(shape的长度)

# 1.tf.reshape：改变view不变content
a = tf.random.normal([4, 28, 28, 3])
print(a.shape, a.ndim)
b = tf.reshape(a, [4, 784, 3])
print(b.shape)
c = tf.reshape(a, [4, -1, 3])  # -1 指定的维度由TF自己推导
print(c.shape)
d = tf.reshape(a, [4, 784 * 3])
print(d.shape)
e = tf.reshape(a, [4, -1])
print(e.shape)

print(tf.reshape(tf.reshape(a, [4, -1]), [4, 28, 28, 3]).shape)  # 恢复原来的shape
print(tf.reshape(tf.reshape(a, [4, -1]), [4, 14, 56, 3]).shape)  # 恢复原来的shape
print(tf.reshape(tf.reshape(a, [4, -1]), [4, 1, 784, 3]).shape)  # 恢复原来的shape
# reshape很可能导致潜在的bug，恢复时需要原来的信息
# 最容易出错的是，恢复时将weight和height弄反

# 2.tf.transpose：改变content，即置换维度的位置。
a = tf.random.normal([4, 3, 2, 1])
print(a.shape)
print(tf.transpose(a).shape)  # 默认所有维度都转置
print(tf.transpose(a, perm=[0, 1, 3, 2]).shape)  # 按索引重新排列维度

a = tf.random.normal([4, 28, 28, 3])  # [b,h,w,c]
print(tf.transpose(a, [0, 2, 1, 3]))  # => [b,w,h,c]
print(tf.transpose(a, [0, 3, 2, 1]))  # => [b,c,w,h]
print(tf.transpose(a, [0, 3, 1, 2]))  # => [b,c,h,w]

# 3.tf.expand_dims:增加shape==1的维度
a = tf.random.normal([4, 35, 8])
print(tf.expand_dims(a, axis=0).shape)  # 若指定axis存在且为正数，则在前面插入维度 (1, 4, 35, 8)
print(tf.expand_dims(a, axis=3).shape)  # 若指定axis不存在则就地插入 (4, 35, 8, 1)
print(tf.expand_dims(a, axis=-1).shape)  # 若指定axis存在且为负数，则在后面插入维度 (4, 35, 8, 1)
print(tf.expand_dims(a, axis=-4).shape)  # 指定axis不存在则就地插入 (1, 4, 35, 8)

# 4.tf.squeeze()：减少shape==1的维度
print(tf.squeeze(tf.zeros([1, 2, 1, 1, 3])).shape)  # 默认去掉所有 shape == 1的维度
a = tf.zeros([1, 2, 1, 3])
print(tf.squeeze(a, axis=0).shape)  # 指定维度，且该维度必须存在
print(tf.squeeze(a, axis=2).shape)  # 指定维度
print(tf.squeeze(a, axis=-2).shape)  # 指定维度
print(tf.squeeze(a, axis=-4).shape)  # 指定维度
