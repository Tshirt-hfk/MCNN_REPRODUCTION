#!/usr/bin/env python
# -*-coding:utf-8-*-

class ConfigFactory:

    def __init__(self, name='mcnn'):
        self.name = name
        self.batch_size = 1
        self.lr = 0.00001
        self.lr_decay = 0.9
        self.momentum = 0.9
        self.start_iters = 1
        self.total_iters = 100001
        self.max_ckpt_keep = 1
        self.ckpt_router = './ckpts/' + self.name + r'/'
        self.log_router = './logs/' + self.name + r'/'
        self.data_root_dir = 'D:/material/image_datasets/crowd_counting_datasets/ShanghaiTech_Crowd_Counting_Dataset/'

    def display_configs(self):
        msg = '''
        ------------ info of %s model -------------------
        batch size              : %s
        learing rate            : %f
        learing rate decay      : %f
        momentum                : %f
        iter num                : %s
        max ckpt keep           : %s
        ckpt router             : %s
        log router              : %s
        data root router        : %s
        ------------------------------------------------
        ''' % (self.name, self.batch_size, self.lr, self.lr_decay, self.momentum, self.total_iters, self.max_ckpt_keep,
               self.ckpt_router, self.log_router, self.data_root_dir)
        print(msg)
        return msg



if __name__ == '__main__':
    configs = ConfigFactory()
    configs.display_configs()
