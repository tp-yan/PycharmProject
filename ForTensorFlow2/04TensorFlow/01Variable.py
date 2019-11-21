import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a=tf.range(5)
print(a)
# Variable包装了Tensor，本质上还是Tensor，但是具有其他属性，使其可训练可求导
b = tf.Variable(a)
print(b.dtype)
print(b.name)
print(b.trainable)

b = tf.Variable(a,name="input_data")
print(b.name)
print(b.trainable)

print(isinstance(b,tf.Tensor))
print(isinstance(b,tf.Variable))
print(tf.is_tensor(b))
bb = b.numpy()
print(type(bb),bb)