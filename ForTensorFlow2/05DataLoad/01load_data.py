import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1. load minist
def load_mnist():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    print(x.shape, y.shape)
    print(x_test.shape, y_test.shape)
    print(x.min(), x.max(), x.mean())  # 因为x还是Numpy类型，故有这些方法，而在TF中对应是 reduce_min/max/mean
    print(y[:4])
    y_onehot = tf.one_hot(y, depth=10)
    print(y_onehot[:2])


load_mnist()


# 2. load cifar
def load_cifar():
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()  # 换成cifar100加载 cifar100数据集
    print(x.shape, y.shape)
    print(x_test.shape, y_test.shape)
    print(x.min(), x.max())
    print(y[:4])
    return (x, y), (x_test, y_test)


(x, y), (x_test, y_test) = load_cifar()


# 3.  from_tensor_slices():Tensor to Dataset
def tensor_to_Dataset(data):
    print("==========tensor_to_Dataset=============")
    # data可以是单个Tensor，也可以是tuple/list包含多个Tensor
    db = tf.data.Dataset.from_tensor_slices(data)
    iter_db = iter(db)
    if isinstance(data, list) or isinstance(data, tuple):
        sample = next(iter_db)
        print(sample[0].shape, sample[1].shape)
    else:
        print(next(iter_db).shape)
    return db


# numpy to tensor 并进行类型转换和一点预处理(缩放)
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
tensor_to_Dataset(x)
tensor_to_Dataset((x, y))


# 4.Dataset.shuffle
def func_shuffle(data, buffer):
    db = tf.data.Dataset.from_tensor_slices(data)
    db = db.shuffle(buffer)  # buffer:将每buffer大的一段数据进行打乱
    return db


db = func_shuffle((x_test, y_test), 10000)  # Dataset 对象


# 5. Dataset.map：常用于进行预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.squeeze(y, axis=0)  # cifar数据集标签会多出一个1维度，直接去掉即可
    y = tf.one_hot(y, depth=10)
    return x, y


def dataset_map(db):
    print("==========preprocess=============")
    db2 = db.map(preprocess)
    res = next(iter(db2))
    print(res[0].shape, res[1].shape)
    print(res[1][:2])


dataset_map(db)


# 6. Dataset.batch
def dataset_batch(db):
    print("==========dataset_batch=============")
    db = db.batch(32)
    res = next(iter(db))  # 原来一次返回一个样本。现在一次返回一批32个样本
    print(res[0].shape, res[1].shape)


dataset_batch(db)


# 7. Dataset.repeat()
def dataset_repeat(db):
    db1 = db.repeat()  # 重复利用数据集，用于循环永不停止
    db2 = db.repeat(2)  # 重复使用数据集2次
    return db2


# 8.数据加载 + 预处理 完整例子
def prepare_mnist_features_and_labels(x, y):
    """
    转换成Tensor格式，并进行部分预处理操作
    :param x: 加载的样本 tensor 格式
    :param y: 加载的标签 tensor 格式
    :return:
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def mnist_dataset():
    print("==========fashion_mnist=============")
    (x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()  # fashion_mnist：分类物体是衣物不是数字
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_val = tf.convert_to_tensor(x_val)
    y_val = tf.convert_to_tensor(y_val)

    dt = tf.data.Dataset.from_tensor_slices((x, y))
    dt = dt.map(prepare_mnist_features_and_labels)
    dt = dt.shuffle(60000).batch(100)  # 将全部样本随机打乱

    dv = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    dv = dv.map(prepare_mnist_features_and_labels)
    dv = dv.shuffle(10000).batch(100)

    return dt, dv


dt, dv = mnist_dataset()
iter_dt, iter_dv = iter(dt), iter(dv)
dt_sample, dv_sample = next(iter_dt), next(iter_dv)
print(dt_sample[0].shape,dt_sample[1].shape) #
print(dv_sample[0].shape,dv_sample[1].shape) #
