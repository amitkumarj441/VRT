from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form
    Args:
        labels: (LongTensor) class labels, sized [#labels,]
        num_classes: (int) number of classes
    Return:
        one_hot_label: (tensor) encoded labels, size [#labels, #classes]
    '''
    one_hot = torch.eye(num_classes)            # [#classes, #classes]
    return one_hot[labels.long()]               # [#labels,  #classes]

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        super(FocalLoss, self).__init__()

    def focal_loss(self, pred, targ):
        '''Focal loss
        Args:
            pred: (tensor) sized [#anchors, #num_classes]
            targ: (tensor) sized [#anchors, ]
        Return:
            loss = -1*alpht*(1-p_t)**gamma*log(p_t)
        '''
        alpha = 0.25
        gamma = 2

        targ = one_hot_embedding(targ.data.cpu(), 1+self.num_classes)
        targ = targ[:,1:]                # exclude_background
        targ = Variable(targ).cuda()    # [#anchors, #num_clsasses]

        p = pred.sigmoid()
        # p_t = p if pred is targ,        p_t = 1-p if pred is not targ
        p_t = p*targ + (1-p)*(1-targ)          
        # coeff = alpha if pred is targ,  coeff = 1-alpha if pred is not targ
        coeff = alpha*targ + (1-alpha)*(1-targ) 
        coeff *= (1-p_t).pow(gamma)

        # within BCEwithlogits, pred and target go through sigmoid function
        return F.binary_cross_entropy_with_logits(pred, targ, coeff, size_average=False)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (locs_preds, loc_targets) and (cls_preds, cls_targets)
        Args:
            loc_preds:      (tensor) [#batch_size, #anchors, 4]
            loc_targets:    (tensor) [#batch_size, #anchors, 4]
            cls_preds:      (tensor) [#batch_size, #anchors, #num_classes]
            cls_targets:    (tensor) [#batch_size, #anchors, #num_classes]
        Return:
            loss: (tensor) smoothL1Loss(loc_preds, loc_targets) 
                + FocalLoss(cls_preds, cls_targets)
        '''
        batch_size, anchor_size = cls_targets.size()
        pos = cls_targets > 0   # [N, #anchors] exclude background
        num_pos = pos.data.long().sum().float()

        mask = pos.unsqueeze(2).expand_as(loc_preds)    # [batch, #anchors, 4] exclude background
        masked_loc_preds = loc_preds[mask].view(-1,4).float()   # [#pos.sum, 4] masked loc_preds coor
        masked_loc_targets = loc_targets[mask].view(-1,4).float()
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        pos_neg = cls_targets > -1   # exclude ignored anchros
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        
        loss = (loc_loss+cls_loss)/num_pos
        return loss, loc_loss/num_pos, cls_loss/num_pos
