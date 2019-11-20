# 只使用Numpy完成线性回归模型的训练，使用CPU完成训练

import numpy as np
import math


# y = wx+b
def compute_cost(b, w, points):
    """
    计算在指定w,b下模型在训练集上的损失值（根据MSE）
    points:所有训练样本
    :return:当前模型在训练集上的损失值
    """
    totalError = 0
    N = len(points)
    for i in range(N):
        x = points[i, 0]  # numpy特有的索引方法 ==> points[i][0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError * 1.0 / N  # 返回均方误差


def step_gradient(b_cur, w_cur, points, lr):
    """
    计算w,b的梯度（在所有样本上）
    :param b_cur:当前b
    :param w_cur:当前w
    :param points:所有样本
    :param lr:学习率
    :return:更新后的b,w:[new_b,new_w]
    """
    N = len(points)
    b_gradient = 0
    w_gradient = 0

    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b-y)
        b_gradient += (2 / N) * (w_cur * x + b_cur - y)
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2 / N) * ((w_cur * x + b_cur - y) * x)
    # update w',b'
    new_b = b_cur - lr * b_gradient
    new_w = w_cur - lr * w_gradient
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, lr, num_iter):
    """
    执行梯度下降法的全过程，在这里控制参数迭代更新的次数
    :param points: 所有样本
    :param starting_b: 初始b
    :param starting_w: 初始w
    :param lr: 学习率
    :param num_iter: 迭代次数
    :return:最后更新的参数[b,w]
    """
    b = starting_b
    w = starting_w
    points = np.array(points)
    # update for several times
    for step in range(num_iter):
        b, w = step_gradient(b, w, points, lr)  # 迭代一次
        # 迭代过程中输出查看loss
        if step % 100 == 0:
            print(compute_cost(b, w, points))
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    lr = 0.0001 # 注意学习率过大会导致模型不收敛，loss会越来越大！！！
    init_b = 0
    init_w = 0
    num_iter = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}".format(init_b, init_w,
                                                                              compute_cost(init_b, init_w, points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, init_b, init_w, lr, num_iter)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iter, b, w, compute_cost(b, w, points)))


if __name__ == "__main__":
    run()
