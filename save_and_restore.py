import numpy as np
import tensorflow  as  tf

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)





saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./tmp'))

graph = tf.get_default_graph()
b2=graph.get_tensor_by_name("biases:0")




# saver = tf.train.Saver()
# save_path = saver.save(sess, './tmp/my-model', global_step=0)
# save_path = saver.save(sess, "./tmp/model.ckpt")
# print("Model saved in file: %s" % save_path)


# saver.restore(sess, "./tmp/model.ckpt")
# print("Model restored.")


print(sess.run('biases:0'))
print(sess.run(b2))

writer = tf.summary.FileWriter('./graphs', sess.graph)
writer.close()
