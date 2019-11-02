# P96 使用tensorflow提供的类来下载、封装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

# 加载MNIST数据集，如果指定地址/path/... 没有数据，则TF从P95表5-1中网址下载
# 返回一个封装好的mnist类，里面包含了 train test validation三个数据集
mnist = input_data.read_data_sets("/path/to/MNIST_data/",not_hot=True)

print("Training data size: ", mnist.train.num_examples)
print("Validating data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)

# 打印一个训练样本数据
# 原图是 28x28=784的二维矩阵数据，但神经网络输入是一个特征向量，故TF将原数据转为 长度为784的一维数组
print("Example training data: ", mnist.train.images[0])

# 打印上面样本对应的标签数据
print("Example trainging data label: ", mnist.train.labels[0])

# 为了方便使用 batch + 随机梯度下降 优化算法，还提供了一下函数，每次读取一个训练batch
batch_szie = 100
xs, ys = mnist.train.next_batch(batch_szie)
print("X shape: ", xs.shape)	# (100,784)
print("Y shape: ", ys.shape)	# (100,10)
