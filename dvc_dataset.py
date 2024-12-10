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
from test_video import PSNR, ms_ssim

# 以下参数来自 DCVC_net.py
out_channel_mv = 128
out_channel_N = 64  
out_channel_M = 96

vimeo_data_path = '' # 'H:/Data/vimeo_septuplet/vimeo_septuplet/sequences/''../../../../../../mnt/h/Data/vimeo_septuplet/vimeo_septuplet/sequences/'
vimeo_test_list_path = '/mnt/data3/zhaojunzhang/vimeo_septuplet/mini_dvc_test_10k.txt'

class DataSet(data.Dataset):
    def __init__(self, path=vimeo_test_list_path, im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir='/mnt/data3/zhaojunzhang/vimeo_septuplet/sequences/', filefolderlist=path)
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
        return input_image, ref_image # , quant_noise_feature, quant_noise_z, quant_noise_mv
        
class RawDataSet(data.Dataset):
    def __init__(self, path=vimeo_test_list_path):
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir='/mnt/data3/zhaojunzhang/vimeo_septuplet/sequences/', filefolderlist=path)
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
    

class UVGDataSet(data.Dataset):
    def __init__(self, root="/mnt/data3/zhaojunzhang/uvg4dcvc/images/", filelist="/mnt/data3/zhaojunzhang/uvg4dcvc/originalv.txt", refdir='', testfull=False, im_height=256, im_width=256):
        self.im_height = im_height
        self.im_width = im_width
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        # self.refbpp = []
        self.input = []
        self.hevcclass = []
        # AllIbpp = self.getbpp(refdir)
        ii = 0
        for folder in folders:
            seq = folder.rstrip()
            # seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                num = i * 12 + 1
                refpath = os.path.join(root, seq, refdir, 'im'+str(num).zfill(3)+'.png')
                inputpath = os.path.join(root, seq, 'im'+str(num+2).zfill(3)+'.png')
                # inputpath = []
                # for j in range(12):
                #     inputpath.append(os.path.join(root, seq, 'im' + str(i * 12 + j + 1).zfill(3)+'.png'))
                self.ref.append(refpath)
                # self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

        # print(self.ref)
        # print(self.input)

    
    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        input_image = imageio.imread(self.input[index])
        ref_image = imageio.imread(self.ref[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)

        return input_image, ref_image
