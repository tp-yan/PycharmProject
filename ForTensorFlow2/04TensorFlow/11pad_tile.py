import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.pad: 默认填充0
a = tf.reshape(tf.range(9), [3, 3])
print(tf.pad(a, [[0, 0], [0, 0]]))  # 行上下、列左右都不填充
print(tf.pad(a, [[1, 0], [0, 0]]))  # 行上填充一行
print(tf.pad(a, [[1, 1], [0, 0]]))  # 行上下各填充一行
print(tf.pad(a, [[1, 1], [2, 0]]))  # 行上下各填充一行，列左2行
print(tf.pad(a, [[1, 1], [1, 1]]))  # 上下左右各一行

# apply image padding
a = tf.random.normal([4, 28, 28, 3])
b = tf.pad(a, [[0, 0], [2, 2], [2, 2], [0, 0]])  # 每个维度都需要指明填充多少
print(b.shape)

# 2.tile
a = tf.reshape(tf.range(6), [3, 2])
print(tf.tile(a, [2, 1]))  # 需要指定每个维度的复制次数，1：保持原来不复制
print(tf.tile(a, [3, 2]))  # 先复制小维度，再复制大维度

a = tf.reshape(tf.range(9), [3, 3])
aa = tf.expand_dims(a, axis=0)
print(aa)
print(tf.tile(aa, [2, 1, 1]))
print(tf.broadcast_to(aa, [2, 3, 3]))
