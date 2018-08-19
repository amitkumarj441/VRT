from retinanet import *

def relation_net():
    def __init__():
        loc_labels, class_label = retinanet()
        features = tf.reshape()
        
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
           output_vgg16, _ = vgg16(input_image, is_training=True)
        
        
        

