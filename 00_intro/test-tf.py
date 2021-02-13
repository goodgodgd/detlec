import tensorflow as tf
x = tf.random.uniform((2, 3, 4))
print(tf.reduce_sum(x, axis=(1, 2)))
