import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.数据准备
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()  # 在线下载MNIST数据集。返回Numpy array格式
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # Numpy array格式转为Tensor格式才能被TF操作
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=10)  # 原始标签是0~9，进行loss计算时，基于one-hot编码
print(x_train.shape, y_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 将Tensor数据集转为Dataset类型方便按批划分、提取训练集
train_dataset = train_dataset.batch(200)  # train_dataset默认批大小是1，这里设为一批200张手写图片

# 2.搭建模型
model = keras.Sequential([
    layers.Dense(512, activation='relu'),  # Dense：全连接层，设置节点数即输出大小以及激活函数
    layers.Dense(256, activation='relu'),
    layers.Dense(10)]) # 没有使用sigmoid，直接使用输出数值进行大小比较。

# 3.设置优化器
optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    """
    迭代一次训练集，分为若干批次训练。
    :param epoch: 当前epoch数
    :return:
    """
    # 一次训练一批。step:batch id
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))  # 平展为一维向量. 相当于输入层 784 个节点
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)  # 不用管中间过程
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]  # 这批次样本的损失值MSE
            # Step3. optimize and update w1, w2, w3, b1, b2, b3。根据链式法则自动求模型参数的梯度
            grads = tape.gradient(loss, model.trainable_variables)
            # w' = w - lr * grad。用梯度更新当前参数
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 打印当前loss
            if step % 100 == 0:
                print(epoch, step, "loss:", loss, loss.numpy())  # loss:Tensor类型。loss.numpy()：转为numpy数据类型
                print("out:",out.shape,out[0])


def train():
    for epoch in range(10):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
