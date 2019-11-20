
import tensorflow as tf

x = tf.constant(1.0)
a = tf.constant(2.0)
b = tf.constant(3.0)
c = tf.constant(4.0)

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a**2 *x +b*x +c
    