import tensorflow  as  tf


# g = tf.Graph()
session = tf.Session()
#
#
# print('get_context_session%s' % tf.get_default_session())
# with tf.Graph().as_default() as gg:
#     with tf.Session().as_default() as sess:
#
#         # print(sess)
#         # print(session)
#         print('global_graph%s'%session.graph)
#         print("gg%s"%sess.graph)
#         c1 = tf.constant(30.0,name='c1')
#     print('gg%s'%gg)
#     print('gg%s'%c1.graph)
#     print('g%s'%g)
#     assert c1.graph is gg
# assert 1 is 1
# with g.as_default() as kk:
#     with session.as_default():
#         print('global_graph%s'%session.graph)
#         c2=tf.constant(333,name='c2')
#         v3 = tf.get_variable('v3', [2])
#     print('g%s'%c2.graph)
#     print('name%s'%c2.op.name)
#     print('g%s'%kk)
#     print('get_context_graph%s'%tf.get_default_graph())
# print('global_graph{0}'.format(tf.constant(1).graph))
# print(c1)
#
# c3=tf.constant(222,name='c3')
# tf.summary.scalar('c1', c1)
# tf.summary.scalar('c2', c2)
# tf.summary.scalar('c3', c3)
# merged = tf.summary.merge_all()
#
#
# vv=tf.Variable(3,name='vv')
# v=tf.get_variable('v',[2,3])
# with tf.variable_scope('vvv'):
#     v1=tf.get_variable('v1',[2])
#     v2=tf.get_variable('v2',[2])
#
#
# print(tf.get_collection('c3_1:0'))
# print(tf.get_collection(tf.GraphKeys.SUMMARIES))
# print(tf.GraphKeys.TRAINABLE_VARIABLES)
# print(tf.Graph.get_all_collection_keys())


# session.run(tf.global_variables_initializer())

# tf.train.write_graph(session.graph, './tmp/my-model', 'train.pbtxt', as_text=True)
# writer = tf.summary.FileWriter('./tmp/my-model', session.graph,max_queue=1)
#
# summary =session.run(merged)
# writer.add_summary(summary)
# writer.close()



# with tf.Session() as sess:
# 	print(c.eval()) # >> 10
# 	sess.run(c)
# W = tf.Variable(tf.truncated_normal([700, 10]))
# with tf.Session() as sess:
# 	sess.run(W.initializer)
# 	print(W.eval())

# inputs = tf.constant(...)
# with g.name_scope('my_layer') as scope:
#     weights = tf.Variable(..., name="weights")
#     biases = tf.Variable(..., name="biases")
#     affine = tf.matmul(inputs, weights) + biases
#     output = tf.nn.relu(affine, name=scope)

#
# # 1. Using Graph.as_default():
# g = tf.Graph()
# with g.as_default():
#     c = tf.constant(5.0)
#     assert c.graph is g
#
# # 2. Constructing and making default:
# with tf.Graph().as_default() as g:
#     c = tf.constant(5.0)
#     assert c.graph is g
#



# # 1. Using the Session object directly:
# sess = ...
# c = tf.constant(5.0)
# sess.run(c)
#
# # 2. Using default_session():
# sess = ...
# with ops.default_session(sess):
#     c = tf.constant(5.0)
#     result = c.eval()
#
# # 3. Overriding default_session():
# sess = ...
# with ops.default_session(sess):
#     c = tf.constant(5.0)
#     with ops.default_session(...):
#         c.eval(session=sess)




#
# with g.as_default() as g:
#     c = tf.constant(5.0, name="c")
#     assert c.op.name == "c"
#     c_1 = tf.constant(6.0, name="c")
#     assert c_1.op.name == "c_1"
#
#     # Creates a scope called "nested"
#     with g.name_scope("nested") as scope:
#         nested_c = tf.constant(10.0, name="c")
#         assert nested_c.op.name == "nested/c"
#
#         # Creates a nested scope called "inner".
#         with g.name_scope("inner"):
#             nested_inner_c = tf.constant(20.0, name="c")
#             assert nested_inner_c.op.name == "nested/inner/c"
#         # Creates a nested scope called "inner".
#         with g.name_scope("inner"):
#             nested_inner_c = tf.constant(20.0, name="c")
#             assert nested_inner_c.op.name == "nested/inner_1/c"
#
#         # Create a nested scope called "inner_1".
#         with g.name_scope("inner"):
#             nested_inner_1_c = tf.constant(30.0, name="c")
#             assert nested_inner_1_c.op.name == "nested/inner_2/c"
#
#             # Treats `scope` as an absolute name scope, and
#             # switches to the "nested/" scope.
#             with g.name_scope(scope):
#                 nested_d = tf.constant(40.0, name="d")
#                 assert nested_d.op.name == "nested/d"
#
#                 with g.name_scope(""):
#                     e = tf.constant(50.0, name="e")
#                     assert e.op.name == "e"
