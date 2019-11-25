import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Selective Indexing：选择索引。按照给定索引重新选择及其顺序
a = tf.random.normal([4,35,8])
# 1.tf.gather()：只在一个维度上做选择索引
print(tf.gather(a,axis=0,indices=[2,3]).shape) # axis=0（默认）在第一个维度上进行索引选择。这里选择第一维上的第2,3个元素
print(a[2:4].shape)
print(tf.gather(a,axis=0,indices=[2,1,3,0]).shape) # 重组样本顺序。indices可选范围 0~3
print(tf.gather(a,axis=1,indices=[2,7,3,16,21]).shape) # indices可选范围 0~34
print(tf.gather(a,axis=2,indices=[2,7,6,5]).shape) # indices可选范围 0~7

# 2.tf.gather_nd():在多个维度上做选择索引
# 若在多个维度下进行选择索引，使用`tf.gather`则需要多次才能完成，更方便的是使用tf.gather_nd()一次完成
print('==============================')
a = tf.random.normal([4,35,8])
print(tf.gather_nd(a,[0]).shape) # 选择第一维的0元素，其他全选
print(tf.gather_nd(a,[0,1]).shape) # 选择第一维的0元素和第二维的1元素，其他全选
print(tf.gather_nd(a,[0,1,2]).shape) # 返回结果是标量
print(tf.gather_nd(a,[[0,1,2]]).shape) # 对返回结果标量外面套了一个[]，使其变向量

# 将最里层的元素看作是一个索引坐标，若存在多个则将多个索引结果按列方向拼接
print(tf.gather_nd(a,[[0,0],[1,1]]).shape) # (2, 8)
print(tf.gather_nd(a,[[0,0],[1,1],[2,2]]).shape)
print(tf.gather_nd(a,[[0,0,0],[1,1,1],[2,2,2]]).shape) # (3,)
print(tf.gather_nd(a,[[[0,0,0],[1,1,1],[2,2,2]]]).shape) # (1, 3)
# recommended indices format: [[0], [1],…]  [[0,0], [1,1],…] [[0,0,0], [1,1,1],…]，即一般2个[]，不多于3个[]

# 3. tf.boolean_mask()：根据bool索引来选择
a = tf.random.normal([4,28,28,3])
print(tf.boolean_mask(a,mask=[True,True,False,False]).shape) # 默认axis=0，mask长度需与选择的维度长度一致.
print(tf.boolean_mask(a,mask=[True,True,False],axis=3).shape) # mask:(3,)
# 多维度选择
a = tf.ones([2,3,4])
# [True,False,False]：索引坐标。第一维的第0个元素，选择第二维的第一个元素，其余不选择，故 [4]
# [False,True,True,]：索引坐标。第一维的第1元素，选择第二维的第2,3元素，故 [4][4]
# 最后结果拼接 3x[4]
print(tf.boolean_mask(a,mask=[[True,False,False],[False,True,True]])) # mask:2x3