import numpy as np
import tensorflow as tf

'''
prediction = tf.nn.softmax(...)  # Output of neural network
label = tf.placeholder(tf.float32, [100, 10])
cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)

# entorpy = -tf.reduce_sum(p*tf.log(p),axis=1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_x, batch_label = data.next_batch()
    sess.run(train_step, feed_dict={x: batch_x,
                                label: batch_label})
'''
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
# if 0==0:
#     raise ValueError('sdasdasdas')
#
#
# tf.app.flags.DEFINE_string("ps_hosts", "",
#                            "Comma-separated list of hostname:port pairs")
# FLAGS = tf.app.flags.FLAGS
# print(FLAGS.ps_hosts)

# # 'x' is [[1, 1, 1]
# #         [1, 1, 1]]
# tf.reduce_sum(x) ==> 6
# tf.reduce_sum(x, 0) ==> [2, 2, 2]
# tf.reduce_sum(x, 1) ==> [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
# tf.reduce_sum(x, [0, 1]) ==> 6


# # 'x' is [[1., 1.]
# #         [2., 2.]]
# tf.reduce_mean(x) ==> 1.5
# tf.reduce_mean(x, 0) ==> [1.5, 1.5]
# tf.reduce_mean(x, 1) ==> [1.,  2.]


#  # tensor `a` is [1.8, 2.2], dtype=tf.float
#   tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32

# tf.sequence_mask([1, 3, 2], 5) =
# [[True, False, False, False, False],
#  [True, True, True, False, False],
#  [True, True, False, False, False]]

# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
# shape(squeeze(t)) == > [2, 3]
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
# shape(squeeze(t, [2, 4])) == > [1, 2, 3, 1]

# 't' is a tensor of shape [2]
# shape(expand_dims(t, 0)) == > [1, 2]
# shape(expand_dims(t, 1)) == > [2, 1]
# shape(expand_dims(t, -1)) == > [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
# shape(expand_dims(t2, 0)) == > [1, 2, 3, 5]
# shape(expand_dims(t2, 2)) == > [2, 3, 1, 5]
# shape(expand_dims(t2, 3)) == > [2, 3, 5, 1]

#
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat([t1, t2], 0) == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# tf.concat([t1, t2], 1) == > [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
#
# # tensor t3 with shape [2, 3]
# # tensor t4 with shape [2, 3]
# tf.shape(tf.concat([t3, t4], 0)) == > [4, 3]
# tf.shape(tf.concat([t3, t4], 1)) == > [2, 6]
# tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)


# tf.stack(tensors, axis=axis)
# # 'x' is [1, 4]
# # 'y' is [2, 5]
# # 'z' is [3, 6]
# stack([x, y, z]) = > [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
# stack([x, y, z], axis=1) = > [[1, 2, 3], [4, 5, 6]]
# This is the opposite of unstack.  The numpy equivalent is
# tf.stack([x, y, z]) = np.asarray([x, y, z])


# if tf.gfile.Exists(path):
#     with tf.gfile.Open(path) as f:
# if not tf.gfile.Exists(FLAGS.eval_log_dir):
#     tf.gfile.MakeDirs(FLAGS.eval_log_dir)
# tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
# if not tf.gfile.IsDirectory(eval_dir):


# tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]

# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
# tf.slice(input, [1, 0, 0], [1, 1, 3]) == > [[[3, 3, 3]]]
# tf.slice(input, [1, 0, 0], [1, 2, 3]) == > [[[3, 3, 3],
#                                              [4, 4, 4]]]
# tf.slice(input, [1, 0, 0], [2, 1, 3]) == > [[[3, 3, 3]],
#                                             [[5, 5, 5]]]

# Creates a session.
# sess = tf.Session()
# Initializes the variable.
# v=global_step_tensor.initialized_value()
# print(global_step_tensor.initial_value)
with tf.Session().as_default() as sess:
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    sess.run(tf.global_variables_initializer())
    #     # strip leading and trailing 2 elements
    #     foo = tf.constant([1, 2, 3, 4, 5, 6])
    #     print(foo[2:-2].eval())  # => [3,4]
    #
    #     # skip every row and reverse every column
    #     foo = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #     print(foo[::2, ::-1].eval())  # => [[3,2,1], [9,8,7]]
    #
    #     # Insert another dimension
    #     foo = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #     print(foo[tf.newaxis, :, :].eval())  # => [[[3,2,1], [9,8,7]]]
    #     print(foo[:, tf.newaxis, :].eval())  # => [[[3,2,1]], [[9,8,7]]]
    #     print(foo[:, :, tf.newaxis].eval())  # => [[[3],[2],[1]], [[9],[8],[7]]]
    #
    #     # Ellipses (3 equivalent operations)
    #     print(foo[tf.newaxis, :, :].eval())  # => [[[3,2,1], [9,8,7]]]
    #     print(foo[tf.newaxis, ...].eval())  # => [[[3,2,1], [9,8,7]]]
    #     print(foo[tf.newaxis].eval())  # => [[[3,2,1], [9,8,7]]]
    # print(tf.reshape(tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32),[2,3]))
    # print(tf.convert_to_tensor([1, 2, 3, 4, 5, 6], dtype=tf.float32))
    for i in range(10):
        sess.run(global_step_tensor)
        print('global_step: %s' % tf.train.global_step(sess,global_step_tensor ))
        # pass

# input_tensor = np.array([1,2,3,4])
# c=tf.zeros([2,2,2,2,2], tf.int32)
# d=tf.zeros_like(input_tensor)
# e=tf.ones([2, 3], tf.int32)
# f=tf.ones_like(input_tensor)
# g=tf.ones([2, 3], 8)
# Create a tensor [0, 1, 2, 3, 4 ,...]
x = tf.range(1, 10, name="x")
print(x.eval(session=session))

cc = tf.constant(1, shape=[5, 11])

split0, split1, split2 = tf.split(cc, [1, 7, 3], 1)  # 1+7+3=11
print(split0.eval(session=session))
print(split1.eval(session=session))
print(split2.eval(session=session))

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tt1=tf.constant(1,shape=[64,64])
tt2=tf.constant(1,shape=[64,64])
tt3=tf.constant(1,shape=[64,64])
tt4=tt1+tt2+tt3
aaa=list()
for _ in range(11):
    aaa.append(tt1)
print(aaa)
with tf.Session().as_default() as sess:
    # print(tt3.eval(session=sess))
    # print(tt4.eval(session=sess))
    # print(tt4.eval(session=sess).shape)
    print(tf.concat(aaa,0).eval().shape)

# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
# tf.slice(input, [1, 0, 0], [1, 1, 3]) == > [[[3, 3, 3]]]
# tf.slice(input, [1, 0, 0], [1, 2, 3]) == > [[[3, 3, 3],
#                                              [4, 4, 4]]]
# tf.slice(input, [1, 0, 0], [2, 1, 3]) == > [[[3, 3, 3]],
#                                             [[5, 5, 5]]]
