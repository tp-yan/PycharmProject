
import tensorflow as tf
import timeit

with tf.device("/cpu:0"):
    cpu_a = tf.random.normal([10000,1000])
    cpu_b = tf.random.normal([1000,2000])
    print(cpu_a.device,cpu_b.device)

with tf.device("/gpu:0"):# 指定变量声明（存储）设备
    gpu_a = tf.random.normal([10000,1000])
    gpu_b = tf.random.normal([1000,2000])
    print(gpu_a.device,gpu_b.device)

def cpu_run():
    with tf.device("/cpu:0"):
        c = tf.matmul(cpu_a,cpu_b)
    return c

def gpu_run():
    with tf.device("/gpu:0"): # 指定运行设备
        c = tf.matmul(gpu_a,gpu_b)
    return c

# warm up
cpu_time = timeit.timeit(cpu_run,number=10) # cpu_run运行10次，查看平均运行时间
gpu_time = timeit.timeit(gpu_run,number=10)
print("warm up:",cpu_time,gpu_time)

cpu_time = timeit.timeit(cpu_run,number=10) # cpu_run运行10次，查看平均运行时间
gpu_time = timeit.timeit(gpu_run,number=10)
print("run time:",cpu_time,gpu_time)