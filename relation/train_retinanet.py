import pdb
import time
import argparse
import os

import tensorflow as tf

from retinanet import *   # Import model 
from focal_loss import *
from logger import Logger
from encoder import DataEncoder

train_records=[]
test_records=[]
for i in range(100):
    train_records.append('train_retinanet'+str(i)+'.tfrecords')

for i in range(100, 120):
    test_records.append('train_retinanet'+str(i)+'.tfrecords')

def read_record(records, image_size=608, batch_size=8):
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
    images, labels = tf.train.batch([image, label], batch_size=batch_size, capacity=30, num_threads=1)
    return images, labels


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
    lr_feed = 0.001
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

    
    with tf.Session() as sess:

        input_image = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
        label_class = tf.placeholder(tf.float32, [num_feature_maps, batch_size, anchor_num, num_classes])
        label_loc = tf.placeholder(tf.float32, [num_feature_maps, batch_size, anchor_num, 4])

        # setting network
        net = RetinaNet(input_image)
        pred_loc, pred_class = net.output

        if use_pretrained:
            checkpoint_path = 'resnet_v2_101.ckpt'
            saver = tf.train.Saver(tf.global_variables)
            saver.restore(sess, checkpoint_path)
        

        # setting optimizer
        lr = tf.placeholder(tf.float32)
        if opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(lr)
        elif opt == 'SGD':
            optimizer = tf.train.MomentumOptimizer(lr, momentum=momentum)
        else: print('==>>wrong opt name')

        # setting loss
        f_loss = focal_loss(label_class, pred_class)
        r_loss = regression_loss(label_loc, pred_loc)
        loss = f_loss + r_loss
        op = optimizer.minimize(loss)

        # loading exsit weights  
        saver = tf.train.Saver() 
        last_checkpoint = tf.train.latest_checkpoint( traindir, 'checkpoint' )
        if last_checkpoint:
            saver.restore( sess, last_checkpoint )
            print( 'Reuse model form: ', format( last_checkpoint ) )
        else:
            sess.run(init)
            print('no checkpoints found')

        preparer = DataEncoder()

        # train
    
        args.data_name == "OpenImagesDataset"
        print("==>>Loading the data.....", args.data_name)
        train_images,train_labels=read_record(train_records, 608, batch_size)
        test_images,test_labels=read_record(test_records, 608, batch_size)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start_time = time.time()


        for iteration in range(args.start_epoch, total_epoch):
            img, las = sess.run([train_images, train_labels])
            class_labels, location_labels = preparer.prepare_data(las)

            if iteration in decay_epochs:
                lr_feed*=0.5
            
            sess.run(op, feed_dict={input_image: img, label_class: class_labels, label_loc: location_labels, lr: lr_feed})
            
            if iteration % 10 == 0:
                batch_loss = sess.run( [f_loss, r_loss], feed_dict={input_image: img, label_class: class_labels, label_loc: location_labels})
                end_time = time.time()
                
                print('Cost after epoch '+str(epoch)+':  ', 'total_loss:', batch_loss, 'cls_loss:', batch_loss[0], \
                      'loc_loss:', batch_loss[1], 'time_spent:', end_time-start_time )
                info = {'training loss': batch_loss, 'loc_loss': batch_loss[0], \
                    'cls_loss': batch_loss[1]}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

            if iteration % 1000 == 0:
                name = 'retinanet.ckpt'
                saver.save( sess, os.path.join( traindir, name ), global_step = epoch)



if __name__ == '__main__':
    main()
