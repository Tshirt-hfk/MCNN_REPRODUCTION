#!/usr/bin/env python
# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
import os
import random
import time
from networks import multi_column_cnn
from configs import *
from data_loader import *

np.set_printoptions(threshold=np.inf)

def set_gpu(gpu=0):
    """
    the gpu used setting
    :param gpu: gpu id
    :return:
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def train():
    cfig = ConfigFactory()
    #set_gpu(0)
    dataset = 'A'
    # training dataset
    img_root_dir = cfig.data_root_dir + r'part_' + dataset + r'_final/train_data/images/'
    gt_root_dir = cfig.data_root_dir + r'part_' + dataset + r'_final/train_data/ground_truth/'
    # testing dataset
    val_img_root_dir = cfig.data_root_dir + r'part_' + dataset + r'_final/test_data/images/'
    val_gt_root_dir = cfig.data_root_dir + r'part_' + dataset + r'_final/test_data/ground_truth/'

    cfig = ConfigFactory()

    # place holder
    input_img_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))

    # network generation
    inference_density_map = multi_column_cnn(input_img_placeholder)

    # density map loss
    density_map_loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(density_map_placeholder, inference_density_map)))

    # jointly training
    joint_loss = density_map_loss
    # optimizer = tf.train.MomentumOptimizer(configs.learing_rate, momentum=configs.momentum).minimize(joint_loss)
    # adam optimizer
    optimizer = tf.train.AdamOptimizer(cfig.lr).minimize(joint_loss)

    init = tf.global_variables_initializer()


    file_path = cfig.log_router

    # training log route
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # model saver route
    if not os.path.exists(cfig.ckpt_router):
        os.makedirs(cfig.ckpt_router)
    log = open(cfig.log_router + cfig.name + r'_training.logs', mode='a+', encoding='utf-8')
    
    saver = tf.train.Saver(max_to_keep=cfig.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(cfig.ckpt_router)

    # start session
    sess = tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
        print('load model, ckpt.model_checkpoint_path')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    data_loader = ImageDataLoader(img_root_dir, gt_root_dir, shuffle=True, downsample=True, pre_load=True)
    data_loader_val = ImageDataLoader(val_img_root_dir, val_gt_root_dir, shuffle=False, downsample=False, pre_load=True)
    # start training
    for i in range(cfig.start_iters, cfig.total_iters):
        # training
        index = 1
        for blob in data_loader:
            img, gt_dmp, gt_count = blob['data'], blob['gt_density'], blob['crowd_count']
            feed_dict = {input_img_placeholder: (img - 127.5) / 128, density_map_placeholder: gt_dmp}
            _, inf_dmp, loss = sess.run([optimizer, inference_density_map, joint_loss], feed_dict=feed_dict)
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            format_str = 'step %d, joint loss=%.5f, inference= %.5f, gt=%d'
            log_line = format_time, blob['fname'], format_str % (i * data_loader.num_samples + index, loss, inf_dmp.sum(), gt_count)
            log.writelines(str(log_line) + '\n')
            print(log_line)
            index = index + 1

        if i % 50 == 0:
            val_log = open(cfig.log_router + cfig.name + r'_validating_' + str(i) +  '_.logs', mode='w', encoding='utf-8')
            absolute_error = 0.0
            square_error = 0.0
            file_index = 1
            for blob in data_loader_val:
                img, gt_dmp, gt_count = blob['data'], blob['gt_density'], blob['crowd_count']
                feed_dict = {input_img_placeholder: (img - 127.5) / 128, density_map_placeholder: gt_dmp}
                inf_dmp, loss = sess.run([inference_density_map, joint_loss], feed_dict=feed_dict)
                format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                format_str = 'step %d, joint loss=%.5f, inference= %.5f, gt=%d'
                absolute_error = absolute_error + np.abs(np.subtract(gt_count, inf_dmp.sum())).mean()
                square_error = square_error + np.power(np.subtract(gt_count, inf_dmp.sum()), 2).mean()
                log_line = format_time, blob['fname'], format_str % (file_index, loss, inf_dmp.sum(), gt_count)
                val_log.writelines(str(log_line) + '\n')
                print(log_line)
                file_index = file_index + 1
            mae = absolute_error / data_loader_val.num_samples
            rmse = np.sqrt(square_error / data_loader_val.num_samples)
            val_log.writelines(str('MAE_' + str(mae) + '_MSE_' + str(rmse)) + '\n')
            val_log.close()
            print(str('MAE_' +str(mae) + '_MSE_' + str(rmse)))
            saver.save(sess, cfig.ckpt_router + '/v1', global_step=i+1)


if __name__ == '__main__':
    train()
