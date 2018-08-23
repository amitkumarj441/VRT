import tensorflow as tf

def texture_loss(pred_texture, labels_texture):
     return tf.reduce_sum(tf.sigmoid_cross_entropy_with_logits(logits=pre_texture, labels=labels_texture))


def relation_loss(pred_relations, labels_relations):
     return tf.reduce_sum(tf.sigmoid_cross_entropy_with_logits(logits=pre_relations, labels=labels_relations))
