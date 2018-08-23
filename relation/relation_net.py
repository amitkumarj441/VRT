from retinanet import *

from roi_pooling.roi_pooling_ops import roi_pooling

def relation_net(locations, class_labels, relation_numbers=9, texture_num=5, use_obj=True, use_so=False):
    
    relationship_location = get_relationship_location(locations)

    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
       last_conv = vgg16(input_image, is_training=True)

    with tf.variable_scope('roi_pooling'):
        
        x_u = roi_pooling(last_conv, relationship_location, 4, 4)
        x_u = tf.reshape(x_u, [x_u.get_shape.as_list()[0], -1])
        x_u = tf.nn.relu(tf.slim.fully_connected(x_u, 4096))
        x_u = tf.nn.relu(tf.slim.fully_connected(x_u, 4096))
        x_u = tf.slim.fully_connected(x_u, 256)        

        
        x_so = roi_pooling(last_conv, location)
        x_so = tf.reshape(x_u, [x_so.get_shape.as_list()[0], -1])

        x_so = tf.nn.relu(tf.slim.fully_connected(x_so, 4096))
        x_so = tf.nn.relu(tf.slim.fully_connected(x_so, 4096))
        x_so = tf.slim.fully_connected(x_so, 256)
    with tf.variable_scope('output'):
        emb_texture =tf.slim.fully_connected(class_labels, 256)
        texture = tf.slim.fully_connected(tf.concat([x_so, emb_so], 1), texture_num)
            
    with tf.variable_scope('output'):

        lo = tf.slim.fully_connected(locations, 256)

        if use_obj:
            emb_so =tf.slim.fully_connected(class_labels, 256)

        last = tf.concat([x_u, lo, emb_so], 1)
        relation = tf.slim.fully_connected(last, relation_numbers)
        
    return relation, texture

def get_relationship_location(locations):
    pass
        
        

