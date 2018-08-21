import tensorflow as tf
from resnet import *
from resnet import RetinaNet_FPN101
slim = tf.contrib.slim
class RetinaNet():
    """ RetinaNet defined in Focal loss paper
     See: https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, inputs, sess, num_classes=62, num_anchors=9, scope=None, reuse=None):
           

        self.feature_maps = RetinaNet_FPN101(inputs)

        self._num_classes = num_classes
        self._num_anchors = num_anchors
        self._scope = scope
        self._reuse = reuse
        self.output = self.forward(inputs)

    def add_fcn_head(self, inputs, output_planes, head_offset):
        """
        inputs: a [batch, height, width, channels] float tensor
        output_planes: # of outputs dim
        layer_offset: idx of feature maps
        """
        with tf.variable_scope(self._scope, "Retina_Head_"+str(head_offset), [inputs], reuse=self._reuse):
            net = slim.repeat(inputs, 4, slim.conv2d, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu)
            net = slim.conv2d(net, output_planes, kernel_size=[3, 3], activation_fn=None)
        return net

    def forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        loc_predictions = []
        class_predictions = []
        for idx, feature_map in enumerate(self.feature_maps):
            loc_prediction = self._add_fcn_head(feature_map,
                                                self._num_anchors * 4,
                                                "Box")
            class_prediction = self._add_fcn_head(feature_map,
                                                  self._num_anchors*self._num_classes,
                                                  "Class")
            loc_prediction = tf.reshape(loc_prediction, [batch_size, -1, 4])
            class_prediction = tf.reshape(class_prediction, [batch_size, -1, self._num_classes])
            loc_predictions.append(loc_prediction)
            class_predictions.append(class_prediction)
        return tf.concat(loc_predictions, axis=1), tf.concat(class_predictions, axis=1)
      
