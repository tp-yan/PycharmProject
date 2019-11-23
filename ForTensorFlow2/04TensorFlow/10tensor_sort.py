import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(2467)

# 1.tf.sort：完全排序
# tf.argsort：全排序后的索引顺序
a=tf.random.shuffle(tf.range(5))
print(tf.sort(a,direction="DESCENDING"))
print(tf.sort(a))
idx = tf.argsort(a,direction="DESCENDING")
print(idx)
print("结合gather手动排序：",tf.gather(a,idx)) # 手动排序
# 高维Tensor，默认对最后一个维度完全排序：
a=tf.random.uniform([3,3],maxval=10,dtype=tf.int32)
print(tf.sort(a)) # 默认行方向排序
print(tf.sort(a,direction='DESCENDING'))
idx=tf.argsort(a)
print(idx)

# 2. tf.math.top_k(tensor,k)
# 返回最大的前k个元素(valuse)和其索引(indices)
res = tf.math.top_k(a,2)
print(res.indices)
print(res.values)
# Top-k accuracy example 1：
prob = tf.constant([[0.1,0.2,0.7],[0.1,0.8,0.1]]) # 输出各类概率
target = tf.constant([2,0]) # 真实标签
k_b = tf.math.top_k(prob,3).indices
k_b = tf.transpose(k_b,perm=[1,0]) # 行列置换，即矩阵转置
target=tf.broadcast_to(target,[3,2]) # [2] => [1,2] => [3,2]

def accuracy(output,target,top=(1,)):
    maxk = max(top)
    batch_size = target.shape[0]
    pred = tf.math.top_k(output,maxk).indices  # 只要前k个最大概率索引，即代表预测类别
    pred = tf.transpose(pred,perm=[1,0])
    target_ = tf.broadcast_to(target,pred.shape)
    correct = tf.equal(pred,target_)

    res = []
    # 依次求top1,top2,...,topk
    for k in range(maxk):
        correct_k = tf.reduce_sum(tf.cast(tf.reshape(correct[:k],[-1]),dtype=tf.float32)) # 求1的个数，即True个数
        acc = float(correct_k/batch_size)
        res.append(acc)

    return res

# 模拟求top-k
output = tf.random.normal([10,6]) # 10个样本，输出6个类别
output = tf.math.softmax(output,axis=1) # 归一化每行，即每个样本输出概率归一化和为1
target = tf.random.uniform([10],maxval=6,dtype=tf.int32)
print("prob:",output.numpy())
print("pred:",tf.argmax(output,axis=1).numpy())
print("label:",target.numpy())

acc = accuracy(output,target,top=(1,2,3,4,5,6))
print('top1-6 acc:', acc)