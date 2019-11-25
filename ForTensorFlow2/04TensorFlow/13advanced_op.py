import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. tf.where
# 1.1 tf.where(tensor):其中tensor是bool型的，返回tensor中为True的元素位置。
# 例一，筛选出大于0的元素的2种方式：
# boolean_mask
a = tf.random.normal([3, 3])
mask = a > 0
print(tf.boolean_mask(a, mask))
# where + gather_nd
indices = tf.where(mask)
print(indices)
print(tf.gather_nd(a, indices))
# 1.2 tf.where(cond,A,B):根据bool型的cond，若元素为True则从对应位置Tensor A中选元素否则选B中元素。
A = tf.ones([3, 3])
B = tf.zeros([3, 3])
print(tf.where(mask, A, B))

# 2.2.tf.scatter_nd
# 更新指定坐标上的值。
# shape：必须初始为0。
# indices：指定要更新的位置。
# updates：更新值。
# 一维例子：
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 112])
shape = tf.constant([8])  # 一维向量
print(tf.scatter_nd(indices, updates, shape))
# 二维例子：
indices = tf.constant([[0], [2]])
updates = tf.constant([  # 2x4x4，更新第[0] 和[2]的值，对应 4x4 shape
    [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
])
shape = tf.constant([4, 4, 4])
print(tf.scatter_nd(indices, updates, shape))

# 3.tf.meshgrid
# 并行化生成坐标点。
# 3.1 Numpy实现的CPU版坐标点采样，无法GPU加速：
# .linspace(start,end,采样个数)
import numpy as np


def meshgridByNumpy():
    points = []
    for y in np.linspace(-2, 2, 5):
        for x in np.linspace(-2, 2, 5):
            points.append([x, y])
    return np.array(points)


print(meshgridByNumpy())


def meshgridByTF():
    y = tf.linspace(-2., 2, 5)
    print(y)
    x = tf.linspace(1., 5, 5)
    print(x)
    points_x, points_y = tf.meshgrid(x, y)
    print(points_x.shape)  # (5, 5)
    print(points_y.shape)  # (5, 5)
    print(points_x)
    print(points_y)

    points = tf.stack([points_x, points_y], axis=2)  # 插入一个新维度，2个[5,5]并行化为[5,5,2] ([0,0],[0,0]) ==> ([0,0,0],[0,0,1])

    return points


print(meshgridByTF())

# 4. 绘制函数 z = sin(x) + sin(y) 的等高线
import matplotlib.pyplot as plt


def func(x):
    """
    实现函数 z = sin(x) + sin(y)
    :param x:[b, 2]
    :return:
    """
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])
    return z

# 绘制等高线
def draw_contour():
    x = tf.linspace(0., 2 * 3.14, 500) # start必须是浮点型
    y = tf.linspace(0., 2 * 3.14, 500)
    point_x, point_y = tf.meshgrid(x, y)
    points = tf.stack([point_x, point_y], axis=2)
    print("points:", points.shape)
    z = func(points)
    print("z:", z.shape)

    # 创建绘画窗口
    plt.figure("plot 2d func value")
    plt.imshow(z, origin="lower", interpolation='none')
    plt.colorbar()

    plt.figure('plot 2d func contour')
    plt.contour(point_x, point_y, z)
    plt.colorbar()
    plt.show()

draw_contour()