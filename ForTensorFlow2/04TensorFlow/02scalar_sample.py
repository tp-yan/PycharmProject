import tensorflow as tf
import os

# loss和accuracy是标量的典型例子

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# loss eg.
out = tf.random.uniform([4,10]) # 假设输出
y = tf.range(4) # 假设标签
y = tf.one_hot(y,depth=10) # one-hot 编码
print(y)
loss = tf.keras.losses.mse(y,out) # 使用MSE求损失值
loss = tf.reduce_mean(loss) # 平均loss
print(loss.numpy())