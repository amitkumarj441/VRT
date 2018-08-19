from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def create_one_hot(labels, num_classes, label_index):
     '''Embedding labels to one-hot form
     Args:
         labels: (LongTensor) class labels, sized [#labels,]
         num_classes: (int) number of classes
     Return:
         one_hot_label: (tensor) encoded labels, size [#labels, #classes]
     '''
    return slim.one_hot_encoding(label_indexes, num_classes)
  
# class loss
def focal_loss(onehot_labels, cls_preds,
                            alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute sigmoid focal loss between logits and onehot labels
    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)
    Args:
      onehot_labels: Each row labels[i] must be a valid probability distribution
      cls_preds: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)
    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)

        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        predictions = tf.nn.sigmoid(precise_logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        epsilon = 1e-8
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                     name=name, axis=1)
        return losses

# # bounding box loss
def regression_loss(pred_boxes, gt_boxes):
    """
    Regression loss (Smooth L1 loss: also known as huber loss)
    Args:
        pred_boxes: [# anchors, 4]
        gt_boxes: [# anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [# anchors]
    """
#     I used l2 loss
    loss = tf.reduce_sum(tf.square(pred_boxes-gt_boxes))
    return loss

#     def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
#         '''Compute loss between (locs_preds, loc_targets) and (cls_preds, cls_targets)
#         Args:
#             loc_preds:      (tensor) [#batch_size, #anchors, 4]
#             loc_targets:    (tensor) [#batch_size, #anchors, 4]
#             cls_preds:      (tensor) [#batch_size, #anchors, #num_classes]
#             cls_targets:    (tensor) [#batch_size, #anchors, #num_classes]
#         Return:
#             loss: (tensor) smoothL1Loss(loc_preds, loc_targets) 
#                 + FocalLoss(cls_preds, cls_targets)
#         '''
#         batch_size, anchor_size = cls_targets.size()
#         pos = cls_targets > 0   # [N, #anchors] exclude background
#         num_pos = pos.data.long().sum().float()

#         mask = pos.unsqueeze(2).expand_as(loc_preds)    # [batch, #anchors, 4] exclude background
#         masked_loc_preds = loc_preds[mask].view(-1,4).float()   # [#pos.sum, 4] masked loc_preds coor
#         masked_loc_targets = loc_targets[mask].view(-1,4).float()
#         loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

#         pos_neg = cls_targets > -1   # exclude ignored anchros
#         mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
#         masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
#         cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        
#         loss = (loc_loss+cls_loss)/num_pos
#         return loss, loc_loss/num_pos, cls_loss/num_pos
