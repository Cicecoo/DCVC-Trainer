import os
import random
import torch
import torch.nn.functional as F
import cv2

def load_submodule_params(submodule, whole_module_checkpoint, submodule_name):
    submodule_params = submodule.state_dict()
    whole_module_state_dict = torch.load(whole_module_checkpoint)

    # print("submodule_params", submodule_params.keys())
    # print("whole_module_state_dict", whole_module_state_dict.keys())

    for name, param in submodule_params.items():
        full_name = submodule_name + '.' + name
        print("loading", full_name, "to", name)
        if full_name in whole_module_state_dict:
            submodule_params[name] = whole_module_state_dict[full_name]
        else:
            print("WARNING: could not find", full_name, "for", name, "in checkpoint")

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
    save_folder = "runs/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 检测并按序号新建文件夹 train0, train1, ...
    sub_dir_name = "train"
    # 检查当前目录下是否有 train0, train1, ... 文件夹
    sub_dir_index = 0
    while os.path.exists(os.path.join(save_folder, sub_dir_name + str(sub_dir_index))):
        sub_dir_index += 1
    # 新建文件夹
    sub_dir_name = sub_dir_name + str(sub_dir_index)
    save_folder = os.path.join(save_folder, sub_dir_name)
    os.makedirs(save_folder)

    return save_folder