from retinanet import *


def relation_net():
    def __init__(self, relation_numbers=10, use_obj=True, use_so=False):
        location, class_label = retinanet()
        
        relationship_location = self.get_relationship_location(location)
        features = tf.reshape()
        
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
           last_2d, _ = vgg16(input_image, is_training=True)
       
        with tf.variable_scope('roi_pooling'):
            x_u = roi_pooling(last_2d, relationship_location)
            x_u = tf.reshape(x_u, [x_u.get_shape.as_list()[0], -1])
            x_u = tf.slim.fully_connected(x_u, 4096)
            x_u = tf.slim.fully_connected(x_u, 4096)
            x_u = tf.slim.fully_connected(x_u, 256)        

    #         one object location
            if use_so:
                x_so = roi_pooling(last_2d, location)
                x_so = tf.reshape(x_u, [x_so.get_shape.as_list()[0], -1])

                x_so = tf.slim.fully_connected(x_so, 4096)
                x_so = tf.slim.fully_connected(x_so, 4096)
                x_so = tf.slim.fully_connected(x_so, 256)
                x_u = tf.concat([x_u, x_so], 1)

            lo = tf.slim.fully_connected(location, 256)
            
            if use_obj:
                emb_so =tf.slim.fully_connected(class_label, 256)

        last = tf.concat([x_u, lo, emb_so], 1)
        self.relation = tf.slim.fully_connected(location, relation_numbers)
        
    def get_relationship_location(self, location):
        pass
        
        

