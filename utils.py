import os
import random
import torch
import torch.nn.functional as F
# import cv2
from pytorch_msssim import ms_ssim
import numpy as np
from PIL import Image


# PSNR: peak signal-to-noise ratio 峰值信噪比
def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2) # 均方误差
    psnr = 20 * torch.log10(1 / torch.sqrt(mse)) # 10 * log10(max^2/mse) = 20 * log10(max/sqrt(mse)), normalized: max = 1
    return psnr.item()


def read_frame_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1) # 读图默认 HWC，转为 CHW
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255 # 归一化
    return input_image


def load_submodule_params(submodule, whole_module_checkpoint, submodule_name):
    submodule_params = submodule.state_dict()
    whole_module_state_dict = torch.load(whole_module_checkpoint)

    # print("submodule_params", submodule_params.keys())
    # print("whole_module_state_dict", whole_module_state_dict.keys())

    cnt = 0
    for name, param in submodule_params.items():
        full_name = submodule_name + '.' + name
        if full_name in whole_module_state_dict:
            print("loading", full_name, "to", name)
            submodule_params[name] = whole_module_state_dict[full_name]
            cnt += 1
        else:
            print("WARNING: could not find", full_name, "for", name, "in checkpoint")
    print("loaded", cnt, "params for", submodule_name)
    submodule.load_state_dict(submodule_params)


def load_submodule_params_(submodule, whole_module_state_dict, submodule_name):
    submodule_params = submodule.state_dict()

    # print("whole_module_state_dict", whole_module_state_dict.keys())

    cnt = 0
    for name, param in submodule_params.items():
        full_name = submodule_name + '.' + name
        if full_name in whole_module_state_dict:
            print("loading", full_name, "to", name)
            submodule_params[name] = whole_module_state_dict[full_name]
            cnt += 1
        else:
            print("WARNING: could not find", full_name, "for", name, "in checkpoint")
    print("loaded", cnt, "params for", submodule_name)
    submodule.load_state_dict(submodule_params)



def freeze_submodule(submodule_list):
    for submodule in submodule_list:
        for param in submodule.parameters():
            param.requires_grad = False

def unfreeze_submodule(submodule_list):
    for submodule in submodule_list:
        for param in submodule.parameters():
            param.requires_grad = True


def random_crop_and_pad_image(image, size):
    image_shape = image.size()
    # 填充图像，使其至少与 size 一样大
    image_pad = F.pad(image, (0, max(size[1], image_shape[2]) - image_shape[2], 
                              0, max(size[0], image_shape[1]) - image_shape[1]))
    # 随机生成裁剪起始位置
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    # 根据随机偏移量进行裁剪
    image_crop = image_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return image_crop


def random_crop_and_pad_image_list(image_list, size):
    combined = torch.cat(image_list, 0)
    last_image_dim = image_list[0].size()[0]
    image_shape = image_list[0].size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return [combined_crop[i*last_image_dim:(i+1)*last_image_dim, :, :] for i in range(len(image_list))]


def get_save_folder():
    save_folder = "/mnt/data3/zhaojunzhang/runs/dcvc/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 检测并按序号新建文件夹 train0, train1, ...
    sub_dir_name = "train"
    # 检查当前目录下是否有形如 "train + 数字 + 其他" 的文件夹
    sub_dir_index = 0
    while any(os.path.isdir(os.path.join(save_folder, d)) and d.startswith(sub_dir_name + str(sub_dir_index)) 
              for d in os.listdir(save_folder)):
        sub_dir_index += 1
    # 新建文件夹
    sub_dir_name = sub_dir_name + str(sub_dir_index)
    save_folder = os.path.join(save_folder, sub_dir_name)
    os.makedirs(save_folder)

    return save_folder


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip) 