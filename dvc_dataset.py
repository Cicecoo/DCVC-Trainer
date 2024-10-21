'''
参考
https://github.com/ZhihaoHu/PyTorchVideoCompression/blob/master/DVC/dataset.py
'''
import os
import torch
import imageio
import numpy as np
import torch.utils.data as data
# from DVC.subnet.basics import *
# from DVC.subnet.ms_ssim_torch import ms_ssim
from DVC.augmentation import random_flip, random_crop_and_pad_image_and_labels

# 以下参数来自 DCVC_net.py
out_channel_mv = 128
out_channel_N = 64  
out_channel_M = 96

vimeo_data_path = '' # 'H:/Data/vimeo_septuplet/vimeo_septuplet/sequences/''../../../../../../mnt/h/Data/vimeo_septuplet/vimeo_septuplet/sequences/'
vimeo_test_list_path = 'H:/Data/vimeo_septuplet/vimeo_septuplet/mini_dvc_test_10k.txt'

class DataSet(data.Dataset):
    def __init__(self, path=vimeo_test_list_path, im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir='H:/Data/vimeo_septuplet/vimeo_septuplet/sequences/', filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("dataset find image: ", len(self.image_input_list))

    # TODO: 改为使用全部数据？
    def get_vimeo(self, rootdir, filefolderlist):
        with open(filefolderlist) as f:
            data = f.readlines()
            
        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            # print(y)
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            # refnumber = int(y[-5:-4]) - 1
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)

        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv
        
class RawDataSet(data.Dataset):
    def __init__(self, path=vimeo_test_list_path):
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir='H:/Data/vimeo_septuplet/vimeo_septuplet/sequences/', filefolderlist=path)
        print("dataset find image: ", len(self.image_input_list))
        
    def get_vimeo(self, rootdir, filefolderlist):
        with open(filefolderlist) as f:
            data = f.readlines()
            
        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            # print(y)
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            # refnumber = int(y[-5:-4]) - 1
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_input, fns_train_ref
    
    def __len__(self):
        return len(self.image_input_list)
    
    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        return input_image, ref_image