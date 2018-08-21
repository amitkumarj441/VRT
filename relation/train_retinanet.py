import pdb
import time
import argparse
import os
import datasets

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf

from retinanet import *   # Import model 
from focal_loss import focal_loss
from retinanet_utils import freeze_bn
from logger import Logger
from encoder import DataEncoder


def read_record(image_size=608):
    feature = {"label": tf.VarLenFeature(tf.float32), "img_raw": tf.FixedLenFeature([], tf.string)}
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(all_records)
    _, serialized_example = reader.read(filename_queue)
    logging.info('read1')
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    
    image = tf.reshape(image, [image_size, image_size, 3])

    image=tf.cast(image,tf.float32)/255.0
    
    label = features['label']
    logging.info('read2')
    images, labels = tf.train.batch([image, label], batch_size=batch_size, capacity=30, num_threads=1)
    return images, labels


def test(input_image,testloader,device,criterion,logger,step):
    
    for data, loc_targets, cls_targets, _ in testloader:
        inputs = data.to(device)
        loc_preds, cls_preds = model(inputs.cuda())
        loc_preds = tf.concat(loc_preds, 1)
        cls_preds = tf.concat(cls_preds, 1)
        loss, loc_loss, cls_loss  = criterion(loc_preds.float(), loc_targets.cuda(), \
                cls_preds.float(), cls_targets.cuda())
        break
    info = {'test loss': loss.item(), 'test_cls_loss': cls_loss.item(), \
            'test_loc_loss': loc_loss.item()}
    for tag, value in info.items():
        logger.scalar_summary(tag,value,step)
    print('\nTest set: Test_loss: %.5f Test_cls_loss: %.5f Test_loc_loss: %.5f\n'%(loss, \
            cls_loss,loc_loss))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name','-data_name',type=str,default='OpenImages')
    parser.add_argument('--weights','-w',type=str,default='None')
    parser.add_argument('--lr_decay_method','-lrm',type=str,default='retina')
    parser.add_argument('--opt','-opt',type=str,default='SGD')  
    parser.add_argument('--debug','-d',type=str,default='False')
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--num-workers', '-n', type=int, default=os.cpu_count())
    args = parser.parse_args()

    num_workers = os.cpu_count()
    batch_size = 8 
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    gpus = [0, 1]
    is_best = 0
    opt = args.opt
    step = 0
    min_scale = 600
    max_scale = 1000
    image_size = 608
    
    
    train_dir = 'model_path/'

    if args.debug == 'True':
        num_workers = 0 

    transform = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    # dataset
    args.data_name == "OpenImagesDataset":
    trainlist = './data/train'
    testlist = './data/test'
    print("==>>Loading the data.....", args.data_name)
    num_classes = 62 #need to update

    total_iter = 100000.
#     total_epoch = int(total_iter/len(trainloader)*len(gpus))
#     if args.data_name == 'OpenImagesDataset':
#         total_epoch += 30
#     print('==>>Total_epoch size is %d'%(total_epoch))

    lr_decay_method = args.lr_decay_method
    if lr_decay_method == 'luong5':
        start_decay_epoch = int(total_epoch/2)
        decay_times = 5
        remain_epoch = total_epoch - start_decay_epoch
        decay_epochs = [start_decay_epoch]
        for decay_idx in range(decay_times):
            decay_epochs += [int(start_decay_epoch+remain_epoch/decay_times*(decay_idx+1))]
    elif lr_decay_method == 'luong10':
        start_decay_epoch = int(total_epoch/2)
        decay_times = 10
        remain_epoch = total_epoch - start_decay_epoch
        decay_epochs = [start_decay_epoch]
        for decay_idx in range(decay_times):
            decay_epochs += [int(start_decay_epoch+remain_epoch/decay_times*(decay_idx+1))]
    elif lr_decay_method == 'luong234':
        start_decay_epoch = int(total_epoch*2/3)
        decay_times = 4
        remain_epoch = total_epoch - start_decay_epoch
        decay_epochs = [start_decay_epoch]
        for decay_idx in range(decay_times):
            decay_epochs += [int(start_decay_epoch+remain_epoch/decay_times*(decay_idx+1))]
    elif lr_decay_method == 'retina':
        decay_epochs = [int(total_epoch*2/3),int(total_epoch*8/9)]
    decay_idx = 0

    if args.debug == 'True':
        logger = Logger('./logs_debug')
    else:
        logger = Logger('./logs_'+args.data_name+'_'+opt+'_'+lr_decay_method+'_%.5f'%(lr))

    
    input_image = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    label_class = tf.placeholder(tf.float32, [batch_size, anchor_num*num_classes])
    label_loc = tf.placeholder(tf.float32, [batch_size, anchor_num*4])

    # setting network
    pred_class, pred_loc = RetinaNet(num_classes, input_image)

    # setting optimizer
    if opt == 'Adam':
        optimizer=tf.train.AdamOptimizer(lr, weight_decay=weight_decay)
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = tf.train.AdamOptimizer(lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print('==>>wrong opt name')

    # setting loss
    loss = [focal_loss(pred_class, label_class), regression_loss(pred_loc, label_loc)]# nn.CrossEntropyLoss()
    
    op = optimizer.minimize(loss)

    # loading exsit weights
    args.data_name == 'OpenImagesDataset'
    
    saver = tf.train.Saver()
    
    last_checkpoint = tf.train.latest_checkpoint( traindir, 'checkpoint' )
    if last_checkpoint:
         saver.restore( sess, last_checkpoint )
         print( 'Reuse model form: ', format( last_checkpoint ) )
    else:
        sess.run(init)
        print('no checkpoints found')


    # train
    with tf.Session() as sess:
        train_images,train_labels=read_record(image_size)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        start_time = time.time()
        for iteration in range(args.start_iteration, total_iterations):
            img, las = sess.run([train_images, train_labels])
            class_labels, location_labels = prepare_data(las)
            
            sess.run(op, feed_dict={input_image:, label_class:, label_loc:}))
            
            if iteration % 10 == 0:
                batch_loss = sess.run( loss, feed_dict={input_image: img, label_class: class_labels, label_loc: location_labels})
                end_time = time.time()
                
                print('Cost after epoch '+str(epoch)+':  ', 'total_loss:', batch_loss, 'cls_loss:', batch_loss[0], \
                      'loc_loss:', batch_loss[1], 'time_spent:', end_time-start_time )
                info = {'training loss': batch_loss, 'loc_loss': batch_loss[0], \
                    'cls_loss': batch_loss[1]}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)
            if iteration % 10000 == 0:
                test(model, testloader, device, criterion, logger, step)

            if iteration % 1000 == 0:
                name = 'retinanet.ckpt'
                saver.save( sess, os.path.join( traindir, name ), global_step = epoch)

            if epoch == decay_epochs[decay_idx]:

if __name__ == '__main__':
    main()
