import tensorflow as tf
from matplotlib import pyplot as plt

# sess = tf.InteractiveSession()
with tf.Session().as_default() as sess:
    aa = tf.constant(1, shape=[2,3,4])
    bb = tf.expand_dims(aa, -1)
    print(bb.eval(session=sess))
