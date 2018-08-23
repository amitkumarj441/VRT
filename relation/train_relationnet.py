from relation import *
import pdb
import time
import argparse
import os

import tensorflow as tf
from tensorflow import logging
from focal_loss import *
from logger import Logger
import numpy as np

train_records=[]
test_records=[]
for i in range(12):
    train_records.append('gs://detectionchallenge/relationship'+str(i)+'.tfrecords')


def get_batch(train_images, train_labels):
    img, las = sess.run([train_images, train_labels])
    indices = np.array(las.indices)
    indices = np.reshape(indices, [int(np.ceil(len(indices)/11)),11, 2])
    las = np.array(las.values)
    las = np.reshape(las, [int(np.ceil(len(las)/11)), 11])
    batches = []
    temp_list = []
    for i in range(len(las)):
        if i > 0 and indices[i][0][0] != indices[i-1][0][0]:
            batches.append(temp_list)
            temp_list = []
        temp_list.append(las[i])
        if i==len(las)-1:
            batches.append(temp_list)

    cls_labels = []
    loc_labels = []
    relation_labels = []
    for i in range(len(batches)):
        temp = np.array(batches[i])
#                print(temp.shape)
        labels = temp[:, :2].tolist()
        bboxes = temp[:, 2:10].tolist()
        relation = temp[:, 10].tolist()

        loc_trues = loc_trues
        cls_trues = labels
        cls_labels.append(cls_trues)
        loc_labels.append(loc_trues)
        relation_labels.append(relation)
    return cls_labels, loc_labels, relation_labels

def read_record(records, image_size=386, batch_size=8):
    feature = {"label": tf.VarLenFeature(tf.float32), "img_raw": tf.FixedLenFeature([], tf.string)}
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(records)
    _, serialized_example = reader.read(filename_queue)
    logging.info('read1')
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    
    image = tf.reshape(image, [image_size, image_size, 3])

    image=tf.cast(image,tf.float32)/255.0
    
    label = features['label']
    logging.info('read2')
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=30, num_threads=1, min_after_dequeue=10)
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name','-data_name',type=str,default='OpenImages')
    parser.add_argument('--weights','-w',type=str,default='None')
    parser.add_argument('--lr_decay_method','-lrm',type=str,default='retina')
    parser.add_argument('--opt','-opt',type=str,default='Adam')  
    parser.add_argument('--debug','-d',type=str,default='False')
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--num-workers', '-n', type=int, default=os.cpu_count())
    args = parser.parse_args()

    num_workers = os.cpu_count()
    batch_size = 6
    lr_feed = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    gpus = [0, 1]
    is_best = 0
    opt = args.opt
    step = 0
    min_scale = 600
    max_scale = 1000
    image_size = 608
    anchor_num = 9
    
    num_feature_maps = 5

    use_pretrained = False
    
    
    traindir = 'model_path/'

    if args.debug == 'True':
        num_workers = 0 

    num_classes = 62 #need to update

    total_epoch = 100000

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
        logger = Logger('./logs_'+args.data_name+'_'+opt+'_'+lr_decay_method+'_%.5f'%(lr_feed))


    input_image = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    label_class = tf.placeholder(tf.float32, [batch_size, 10, num_classes])
    label_loc = tf.placeholder(tf.float32, [batch_size, 10, 4])
    
    label_relation = tf.placeholder(tf.float32, [batch_size, , 4])

    # setting network
    net = RetinaNet(input_image)
    pred_loc, pred_class = net.output


    # setting optimizer
    lr = tf.placeholder(tf.float32)
    if opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif opt == 'SGD':
        optimizer = tf.train.MomentumOptimizer(lr, momentum=momentum)
    else: print('==>>wrong opt name')

    # setting loss
    op = optimizer.minimize(loss)
    
    with tf.Session() as sess:

        train_images,train_labels=read_record(train_records, image_size, batch_size)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # loading exsit weights  
        saver = tf.train.Saver() 
        last_checkpoint = tf.train.latest_checkpoint( traindir, 'checkpoint' )
        if last_checkpoint:
            saver.restore( sess, last_checkpoint )
            print( 'Reuse model form: ', format( last_checkpoint ) )
        else:
            sess.run(init)
            print('no checkpoints found')

        # train
    
        net = relation_net(label_class, label_loc)
        args.data_name == "OpenImagesDataset"
        print("==>>Loading the data.....", args.data_name)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start_time = time.time()

        for iteration in range(total_epoch):
            start_time = time.time()
            
            cls_labels, loc_labels = get_batch(train_images, train_labels)

            if iteration in decay_epochs:
                lr_feed*=0.5
            if len(loc_labels)!=batch_size or len(cls_labels)!=batch_size:
                continue
            sess.run(op, feed_dict={input_image: img, label_class: cls_labels, label_loc: loc_labels, lr: lr_feed})
            
            if iteration % 10 == 0:
                batch_loss = sess.run(, feed_dict={label_class: cls_labels, label_loc: loc_labels})
                end_time = time.time()
                
                print('Cost after iteration '+str(iteration)+':  ', 'total_loss:', batch_loss, 'cls_loss:', batch_loss[0], \
                      'loc_loss:', batch_loss[1], 'time_spent:', end_time-start_time )
                #info = {'training loss': batch_loss, 'loc_loss': batch_loss[0], \
                 #   'cls_loss': batch_loss[1]}
                #for tag, value in info.items():
                 #   logger.scalar_summary(tag, value, step)
                #lll=encoder.decode(pred_loc[0], pred_class[0])
            if iteration % 1000 == 0:
                name = 'retinanet.ckpt'
                saver.save( sess, os.path.join( traindir, name ), global_step = iteration)

        coord.request_stop()
        coord.join(threads)
        sess.close()
if __name__ == '__main__':
    main()
