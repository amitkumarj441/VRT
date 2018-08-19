from resnet import *
from relation_net import *
slim=tf.contrib.slim

if __name__=='__main__':
   arg_scope = resnet_arg_scope()
   with slim.arg_scope(arg_scope):
    output, _ = resnet_v2_50(input_image, is_training=True)
    
   sess=tf.Session()
   checkpoint_path = 'resnet_v2_101.ckpt'
   saver = tf.train.Saver(tf.global_variables)
   saver.restore(sess, checkpoint_path)
   print(output.shape)
   
   sess.close()
   
