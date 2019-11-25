import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. +,-,*,/,//,%
# Tensor运算要求类型一致
a = tf.fill([2, 2], 2.)  # 不指定类型，默认int32
b = tf.fill([2, 2], 4.)
print(a, b)
print(a + b, a - b, a * b, a / b)
print(b // a, b % a)

# 2. tf.math.log tf.exp
# 注意 log()包位置，并log()只能以e为底
print(tf.math.log(a)) # 参数只能是float，不能是int
print(tf.exp(a))
# 实现log2, log10? 只能用公式实现： 1oga(b) = logc(b) / logc(a)
print("log2(8):", tf.math.log(8.) / tf.math.log(2.))
print("log10(100):", tf.math.log(100.) / tf.math.log(10.))

# 3. tf.pow() == ** 与 tf.sqrt()
print(tf.pow(b,3))
print(b**3)
print(tf.sqrt(b))

# 4.@ == tf.matmul()
print(a@b)
print(tf.matmul(a,b))
# 批量矩阵乘法 [batch,w,h]
a=tf.ones([4,2,3])
b=tf.fill([4,3,5],2.)
# batch [2,3]x[3,5]
print(a@b)
print(tf.matmul(a,b)) # 自动批量化

# 若shape不匹配，则需要借助Broadcasting
b=tf.fill([3,5],2.)
bb=tf.broadcast_to(b,[4,3,5]) # 增加维度
print(a@bb)

# 5. 标量化 ==> 向量化
# Y = X@W + 𝑏
x=tf.ones([4,2])
W=tf.ones([2,1])
b=tf.constant(0.1)
print(x@W+b)
# out = relu(X@W + 𝑏)
out = x@W+b
out = tf.nn.relu(out)
