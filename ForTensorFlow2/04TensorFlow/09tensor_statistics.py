import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.L2-范数
a = tf.ones([2, 2])
print(tf.norm(a))
print(tf.sqrt(tf.reduce_sum(tf.square(a))))
a = tf.ones([4, 28, 28, 3])
print(tf.norm(a))
print(tf.sqrt(tf.reduce_sum(tf.square(a))))

# 2.L1-范数
b = tf.ones([2, 2])
print(tf.norm(b))
print(tf.norm(b, ord=2, axis=1))  # 2范数-行方向，对一行向量求范数
print(tf.norm(b, ord=1))  # 指定为1范数，默认是对所有元素求范数
print(tf.norm(b, ord=1, axis=0))  # 2范数-列方向
print(tf.norm(b, ord=1, axis=1))  # 2范数-行方向

# 3.reduce_min/max/mean
a = tf.random.normal([4, 10])
print(tf.reduce_max(a), tf.reduce_min(a), tf.reduce_sum(a))  # 默认是求全局的最大最小均值
print(tf.reduce_max(a, axis=1), tf.reduce_min(a, axis=0), tf.reduce_sum(a, axis=1))

# 4.argmax/argmin：向量的最大最小值索引
print(tf.argmax(a))  # 默认 axis=0
print(tf.argmax(a, axis=1))
print(tf.argmin(a))
print(tf.argmin(a, axis=1))

# 5.tf.equal：比较对象维度必须一样
a=tf.constant([1,2,3,2,5])
b=tf.range(5)
print(tf.equal(a,b)) # 逐个元素比较
res = tf.equal(a,b)
print(tf.reduce_sum(tf.cast(res,dtype=tf.int32))) # 转为 int，求1的个数，进一步用于求accuracy
print(tf.equal(tf.zeros([2,2]),tf.ones([2,2])))
# 5.1 Accuracy
a=tf.constant([[0.1,0.2,0.7],[0.9,0.05,0.05]])
print(a)
# 预测索引
pred = tf.cast(tf.argmax(a,axis=1),dtype=tf.int32)
y = tf.constant([2,1])
res = tf.equal(pred,y)
correct = tf.reduce_sum(tf.cast(res,dtype=tf.int32))
print(correct)
accuracy = correct / len(y)
print(float(accuracy))

# 6.tf.unique：去除重复元素
a=tf.constant([4,2,2,4,3])
res = tf.unique(a)
print(res)
print(res.y) # 去重结果
idx = res.idx # 包含了原元素在筛选后的结果中的索引，利用idx可以还原原数据
print(idx)
print(tf.gather(res.y,idx)) # 按照 idx索引 对 res.y重新采样

