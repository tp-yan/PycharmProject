import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    """
    预处理mnist数据集
    :param x: Tensor类型
    :param y:
    :return:
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def create_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_db = train_db.shuffle(60000).batch(128).map(preprocess).repeat(30)
    test_db = test_db.shuffle(10000).batch(128).map(preprocess)

    return train_db, test_db


def main(train_db, test_db):
    """
    :param train_db:  Dataset 可迭代对象
    :param test_db:  Dataset
    :return:
    """
    # learning rate
    lr = 1e-3

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1)), tf.Variable(tf.zeros([512]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
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
        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # for g in grads:
        #     print(tf.norm(g))
        # update w' = w - lr*grad
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        if step % 100 == 0:
            print(step, 'loss:', float(loss))

        # evaluate 每500step测一次准确率
        if step % 500 == 0:
            total, total_correct = 0., 0

            for _, (x, y) in enumerate(test_db):
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct / total)


if __name__ == '__main__':
    train_db, test_db = create_dataset()
    main(train_db, test_db)
