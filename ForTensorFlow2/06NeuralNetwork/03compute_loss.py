import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.MSE
y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4)  # [5] => [5,4]
y = tf.cast(y, dtype=tf.float32)

out = tf.random.normal([5, 4])

loss1 = tf.reduce_mean(tf.square(y - out))  # tf.reduce_mean：除以的是所有元素个数。一般是除以batch，但这里还出了向量长度 5x4=20
loss2 = tf.square(tf.norm(y - out)) / (5 * 4)  # L2-norm
loss3 = tf.reduce_mean(tf.losses.MSE(y, out))  # tf.losses.MSE函数:求的是样本的MSE，返回[b]，还需要求该批次的loss均值。注意MeanSquareError是类
# 3个计算结果相同，常用loss1、loss3的计算方式
print(loss1)
print(loss2)
print(loss3)

# 2.Cross Entropy
# 交叉熵用来衡量2个分布的相似度，常作为分类问题的损失函数。
# 2.1 Entropy信息熵计算
# 信息熵：信息可以通过熵来度量。不确定性越大，则信息量越大，熵越大。不确定性越小，则信息量越小，熵越小。
a = tf.fill([4], 0.25)  # 一个分布，一个概率向量（总和为1），描述一件事出现各种状态的可能性
one_item = a * tf.math.log(a) / tf.math.log(2.)
print("entropy one_item:", one_item)  # tf不直接支持求log2，只支持ln，而交叉熵是求log2
a_entropy = -tf.reduce_sum(one_item)

# 熵越大越不稳定，熵越小越稳定
print("entropy a:", a_entropy)
a = tf.constant([0.1, 0.2, 0.1, 0.6])
print("entropy a:", -tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))
a = tf.constant([0.01, 0.02, 0.01, 0.96])
print("entropy a:", -tf.reduce_sum(a * tf.math.log(a) / tf.math.log(2.)))

# 2.2 Cross Entropy计算
# 交叉熵用来衡量2个分布的相似度
# 单样本交叉熵-即loss，值越小说明2者分布越相似
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]))  # 前面是真实标签one-hot编码，后面是预测输出
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.8, 0.0]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.1, 0.1, 0.7, 0.1]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.01, 0.97, 0.01, 0.01]))
# 二分类2个输出节点
print(tf.losses.categorical_crossentropy([0, 1], [0.9, 0.1]))
# 二分类1个输出节点
print(tf.losses.BinaryCrossentropy()([1], [0.1]))  # [1]：标签 [0.1]:输出概率，置信度。BinaryCrossentropy()返回一个实例
print(tf.losses.binary_crossentropy([1], [0.1]))

# 3.数值稳定性
# 从softmax层到cross-entropy层可能会出现数据不稳定，即数值超出有限范围，需要做一些数值优化工作，而以下代码就是将右边阴影框的内容进行封装，
# 并实现了数值优化，只需要传入logits即可，无需自己去实现和softmax层cross-entropy层（不推荐自己实现）：
x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])

logits = x @ w + b  # 直接输出
prob = tf.math.softmax(logits, axis=1)  # 转换为概率
print(tf.losses.categorical_crossentropy([0, 1], logits, from_logits=True))  # from_logits=True 要求输入logits，没有经过激活函数的直接输出
print(tf.losses.categorical_crossentropy([0, 1], prob))  # 不推荐
