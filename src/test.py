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

def test():
    cfig = ConfigFactory()
    #set_gpu(0)
    dataset = 'A'
    # training dataset
    img_root_dir = cfig.data_root_dir + r'part_' + dataset + r'_final/train_data/images/'
    gt_root_dir =  cfig.data_root_dir + r'part_' + dataset + r'_final/train_data/ground_truth/'
    # testing dataset
    val_img_root_dir =  cfig.data_root_dir + r'part_' + dataset + r'_final/test_data/images/'
    val_gt_root_dir =  cfig.data_root_dir + r'part_' + dataset + r'_final/test_data/ground_truth/'

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
    # optimizer = tf.train.AdamOptimizer(cfig.lr).minimize(joint_loss)

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
        print('load model', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    data_loader = ImageDataLoader(img_root_dir, gt_root_dir, shuffle=False, downsample=False, pre_load=False)
    data_loader_val = ImageDataLoader(val_img_root_dir, val_gt_root_dir, shuffle=False, downsample=False, pre_load=False)

    absolute_error = 0.0
    square_error = 0.0
    file_index = 1
    for blob in data_loader_val:
        img, gt_dmp, gt_count = blob['data'], blob['gt_density'], blob['crowd_count']
        feed_dict = {input_img_placeholder: (img - 127.5) / 128, density_map_placeholder: gt_dmp}
        inf_dmp, loss = sess.run([inference_density_map, joint_loss], feed_dict=feed_dict)
        print(blob['fname'], gt_count.sum(), inf_dmp.sum(), loss)
        #print(absolute_error,square_error)
        absolute_error = absolute_error + np.abs(np.subtract(gt_count.sum(), inf_dmp.sum())).mean()
        square_error = square_error + np.power(np.subtract(gt_count.sum(), inf_dmp.sum()), 2).mean()
        file_index = file_index + 1
        show_map(img[0, :, :, 0])
        show_density_map(inf_dmp[0, :, :, 0])
        show_density_map(gt_dmp[0, :, :, 0])
    mae = absolute_error / data_loader_val.num_samples
    rmse = np.sqrt(square_error / data_loader_val.num_samples)
    print(str('MAE_' +str(mae) + '_MSE_' + str(rmse)))

if __name__ == '__main__':
    test()
