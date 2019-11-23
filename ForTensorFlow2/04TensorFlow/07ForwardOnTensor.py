import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只打印Error信息，0：全部打印

# 手动实现前向传播过程
# ================ step1 data prepared ================
# datasets：keras提供的管理数据的工具
# x: [60k, 28, 28],
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
# numpy 2 tensor
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # -1~1的范围更好优化
y = tf.convert_to_tensor(y, dtype=tf.int32)
print(x.shape, x.dtype)
print(y.shape, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 创建数据集，方便一次取一个batch
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
print(train_db)
# 窥探train_db
train_iter = iter(train_db)
print(train_iter)
sample = next(train_iter)  # 拿出其中一个batch训练集，(samples,labels) => ([b,28,28],[b])
print('one batch:', sample[0].shape, sample[1].shape)
# 窥探train_db

# ================ step2 create variables ================
# model : [b, 784] => [b, 256] => [b, 128] => [b, 10]
# 声明可训练参数。必须要有tf.Variable()来包装，否则求变量梯度时，其梯度值不会被保留
# w:[dim_in, dim_out], b:[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256],stddev=0.1)) # 初始值非常重！！若初始权重很大，将导致发散，或者收敛很慢，建议选择很小的初始权重值。
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3  # 学习率

# ================ step3 build and train model ================
for epoch in range(10):  # iterate db for 10。因为每次都是在相同训练集上训练，故最后能达到的优化效果是由上限的
    for step, (x, y) in enumerate(train_db):  # step：第几批batch
        # x:[128, 28, 28]
        # y: [128]
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])  # 因为第一层是全连接层，节点为一维向量，需要预处理图像平展为一维向量输入

        with tf.GradientTape() as tape:  # 自动求解参数梯度的环境
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b, 256] + [b, 256]
            h1 = x @ w1 + b1  # 自动Broadcasting + tf.broadcast_to(b1,[x.shape[0],256])
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2 @ w3 + b3  # 最后一层一般不用或用sigmoid

            # ================ step4 compute loss ================
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)  # ==> loss 矩阵求和再 除以 b*10，变为标量

        # ================ step5 compute variables ================
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # ================ step6 update gradients ================
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])  # 注意这里不能使用 w1 = w1 - lr*grads[0] ，这将导致w1又变成 tensor而非Variable，以后无法记录梯度
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
