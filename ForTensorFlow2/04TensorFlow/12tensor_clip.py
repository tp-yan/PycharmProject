import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.tf.clip_by_value：根据值范围裁剪
a = tf.range(9)
print(a)
print(tf.maximum(a, 2))  # 将a中小于2的截断为2
print(tf.minimum(a, 8))  # 将a中大于8的截断为8
print(tf.clip_by_value(a, 2, 8))  # ==> min(8,max(x,2)) 。直接指定截断范围

# ReLU实现：
a = a - 5
print(" =================== ReLU =================== ")
print(tf.nn.relu(a))
print(tf.maximum(a, 0))

# 2.tf.clip_by_norm：根据范数裁剪张量，即通过设置张量的范数来缩放张量
a=tf.random.normal([2,2],mean=10)
print(a)
print(tf.norm(a)) # 求范数
aa = tf.clip_by_norm(a,15) # 设置 a 的范数为15，让其自动缩放
print(tf.norm(aa))

# 3.Gradient Clipping
# tf.clip_by_norm只能让一个Tensor缩放，而 tf.clip_by_global_norm可以让多个Tensor等比缩放
# tf.clip_by_global_norm常用来缩放梯度，防止梯度消失或爆炸
# new_grads, total_norm = tf.clip_by_global_norm(grads, 25)

from tensorflow import  keras
from tensorflow.keras import datasets,layers,optimizers
print(tf.__version__)

def gradient_exploding_test():
    (x, y), _ = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 50. # 将0~255，缩放到0~5，而不是0~1，更容易出现梯度爆炸问题
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=10)
    print('x:', x.shape, 'y:', y.shape)

    train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128).repeat(30)
    # repeat:数据集重复了指定次数。dataset.repeat就是俗称epoch
    # repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
    x, y = next(iter(train_db))
    print('sample:', x.shape, y.shape)

    # 创建模型参数variable,确保初始值很小
    # 784 => 512
    w1, b1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1)), tf.Variable(tf.zeros([512]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(lr=0.01)

    for step, (x, y) in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:
            # 前向传播
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y - out)
            # [b, 10] => [b]
            loss = tf.reduce_mean(loss, axis=1)
            # [b] => scalar
            loss = tf.reduce_mean(loss)

        # 后向传播+更新
        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        print('==before==')
        for g in grads:
            print(tf.norm(g))

        grads, _ = tf.clip_by_global_norm(grads, 15) # 每次限制更新梯度的范数为15，但是保持梯度方向不变

        print('==after==')
        for g in grads:
            print(tf.norm(g))
        # update w' = w - lr*grad
        optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2, w3, b3]))

        if step % 100 == 0:
            print(step, 'loss:', float(loss))

if __name__ == '__main__':
    gradient_exploding_test()