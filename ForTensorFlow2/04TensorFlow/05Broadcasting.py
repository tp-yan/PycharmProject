import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Broadcasting 步骤与前提： 
1.若维度不齐，则插入维度为1，保证shape相同 
2.从右往左，若同一维度size不相同且没有一个为1，则不能扩张，无法Broadcasting 
3.若能Broadcasting，那么将size小的维度扩充为相等size 
'''
# 1.自动Broadcasting
x = tf.random.normal([4, 32, 32, 3])
print((x + tf.random.normal([3])).shape)
print((x + tf.random.normal([32, 32, 1])).shape)
print((x + tf.random.normal([4, 1, 1, 1])).shape)
# print(x+tf.random.normal([1,4,1,1]).shape) # 第二维度都不为1且不相同不能Broadcasting

# 2.手动Broadcasting：tf.broadcast_to
print(x.shape)
print((x + tf.random.normal([4, 1, 1, 1])).shape)
b = tf.broadcast_to(tf.random.normal([4, 1, 1, 1]), [4, 32, 32, 3])  # 指定要扩充的维度
print(b.shape)

# 3.Broadcast VS Tile
# Broadcasting不会真正消耗内存，而tf.tile()会真实复制一批数据。
a = tf.ones([3, 4])
a1 = tf.broadcast_to(a, [2, 3, 4])
print(a1)
# 使用Tile首先扩充维度，然后复制
a2 = tf.expand_dims(a, axis=0)
a2 = tf.tile(a2, [2, 1, 1])  # 指定每维度复制的次数
print(a2)
