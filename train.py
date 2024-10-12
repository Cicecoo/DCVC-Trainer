# 参考 https://github.com/microsoft/DCVC/issues/35
import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from src.models.DCVC_net import DCVC_net
from src.zoo.image import model_architectures as architectures
from test_video import PSNR, ms_ssim, read_frame_to_torch
from dvc_dataset import DataSet

import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from utils import load_submodule_params, freeze_submodule, unfreeze_submodule

train_args = {
    'i_frame_model_name': "cheng2020-anchor",
    'i_frame_model_path': ["checkpoints/cheng2020-anchor-3-e49be189.pth.tar", 
                           "checkpoints/cheng2020-anchor-4-98b0b468.pth.tar",
                           "checkpoints/cheng2020-anchor-5-23852949.pth.tar",
                           "checkpoints/cheng2020-anchor-6-4c052b1a.pth.tar"],
    'i_frame_model_index': 0,
    'dcvc_model_path': "checkpoints/model_dcvc_quality_0_psnr.pth",
    'test_dataset_config': "dataset_config.json",
    'worker': 1,
    'cuda': True,
    'cuda_device': 0,
    # 'output_json_result_path': "required_value",  
    'model_type': "psnr",
    'resume': False,
    "batch_size": 4,
    "metric": "MSE", # 最小化 MSE 来最大化 PSNR
    "quality": 3,   # 3、4、5、6
    "gop": 10,
    "epochs": 40,
}

# 1.mv warmup; 2.train excluding mv; 3.train excluding mv with bit cost; 4.train all
borders_of_steps = [10, 20, 30]  

# 此处 index 对应文中 quality index
# lambda来自于文中3.4及附录
lambda_set = {
    "MSE": { # 对应psnr
        3: 256, 
        4: 512, 
        5: 1024, 
        6: 2048
    },
    "MS-SSIM": {
        3: 8, 
        4: 16, 
        5: 32, 
        6: 64
    }
}


class Trainer(Module):
    def __init__(self, args):
        super().__init__()
        # 加载 cheng2020-anchor 模型
        i_frame_load_checkpoint = torch.load(
            args['i_frame_model_path'][args['i_frame_model_index']], 
            map_location=torch.device('cpu')
            )
        self.i_frame_net = architectures[args['i_frame_model_name']].from_state_dict(i_frame_load_checkpoint).eval()

        # 加载 DCVC 模型
        self.video_net = DCVC_net()
        if args['resume']:
            load_checkpoint = torch.load(args['dcvc_model_path'], map_location=torch.device('cpu'))
            self.video_net.load_dict(load_checkpoint)
        else:
            # 仅加载光流网络
            load_submodule_params(self.video_net.opticFlow, "checkpoints/model_dcvc_quality_0_psnr.pth", 'opticFlow')

        # 加载到 gpu
        self.device = torch.device("cuda" if args['cuda'] else "cpu")
        self.i_frame_net.to(self.device)
        self.i_frame_net.eval()
        self.video_net.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.video_net.parameters(), lr=1e-4)

        # 超参数
        self.metric = args['metric']
        self.quality_index = args['quality']
        self.gop = args['gop']

        # 初始化
        self.current_epoch = 0
        self.current_step = 0
        self.loss_settings = dict()
        self.freeze_list = [self.video_net.opticFlow,
                            self.video_net.mvEncoder,
                            self.video_net.mvpriorEncoder,
                            self.video_net.mvpriorDecoder, # 是否需要?
                            self.video_net.auto_regressive_mv,
                            self.video_net.entropy_parameters_mv,
                            self.video_net.mvDecoder_part1,
                            self.video_net.mvDecoder_part2,
                            self.video_net.bitEstimator_z_mv
                            ]
    
    def schedule(self):
        if self.current_epoch < borders_of_steps[0]:
            step = 0
            name = "me"
        elif self.current_epoch == borders_of_steps[0]:
            step = 1
            name = "reconstruction"
            # 冻结光流网络
            freeze_submodule(self.freeze_list)
        elif self.current_epoch < borders_of_steps[1]:
            step = 1
            name = "reconstruction"
        elif self.current_epoch == borders_of_steps[1]:
            step = 2
            name = "contextual_coding"
        elif self.current_epoch < borders_of_steps[2]:
            step = 2
            name = "contextual_coding"
        elif self.current_epoch == borders_of_steps[2]:
            step = 3
            name = "all"
            # 解冻光流网络
            unfreeze_submodule(self.freeze_list)
        else:
            step = 3
            name = "all"
        self.current_step = step
        loss_settings = dict()
        loss_settings["step"] = step
        loss_settings["name"] = name

        # 学习率对应原文3.4节
        if step < 3:
            loss_settings["lr"] = 1e-4
        else:
            loss_settings["lr"] = 1e-5
        
        loss_settings["components"] = []
        if step == 0: # 为了 loss 里的循环不再使用此项
            # x_tilde 是 warped frame in pixel domain，见附录B Step1
            # loss_settings["components"].append("x_tilde_dist") 
            pass
        else:
            # loss_settings["components"].append("x_hat_dist")
            pass

        if step == 0: 
            loss_settings["D-item"] = "x_tilde_dist" 
            pass
        else:
            loss_settings["D-item"] = "x_hat_dist"
            pass

        if step == 0 or step == 3:
            loss_settings["components"].append("mv_latent_rate") # gt 
            loss_settings["components"].append("mv_prior_rate") # st
        if step == 2 or step == 3:
            loss_settings["components"].append("frame_latent_rate") # yt
            loss_settings["components"].append("frame_prior_rate") # zt
        # 更新 trainer 的 loss_settings 
        self.loss_settings = loss_settings

        if self.current_epoch == borders_of_steps[2]:
            self.optimizer.param_groups[0]["lr"] = 1e-5
            
    """ 
    return value of DCVC_net.forward
    其中 y 对应 feature，z 对应 latent code（feature 即 prior？）
        {"bpp_mv_y": bpp_mv_y,
         "bpp_mv_z": bpp_mv_z,
         "bpp_y": bpp_y,
         "bpp_z": bpp_z,
         "bpp": bpp,
         "recon_image": recon_image,
         "context": context,
         }

    loss components:
        x_tilde_dist / x_hat_dist: D,
        mv_latent_rate: R(gt),
        mv_prior_rate: R(st),
        frame_latent_rate: R(yt),
        frame_prior_rate: R(zt)
    """
    loss_setting2output_obj = { # 最小化bpp来最大化压缩率
        "mv_latent_rate": "bpp_mv_z",
        "mv_prior_rate": "bpp_mv_y",
        "frame_latent_rate": "bpp_z",
        "frame_prior_rate": "bpp_y"
    }

    def loss(self, net_output, target):
        # warmup 时需要只获取运动补偿输出
        if self.loss_settings["D-item"] == "x_tilde_dist":
            D_item = F.mse_loss(net_output["x_tilde"], target) # x_tilde 取自 DCVC_net 的 motioncompensation
        else:
            D_item = F.mse_loss(net_output["recon_image"], target)
        loss = lambda_set[self.metric][self.quality_index] * D_item
        for component in self.loss_settings["components"]:
            loss += net_output[self.loss_setting2output_obj[component]]
        loss = self.loss_settings["lr"] * loss
        return loss

    def training_step(self, batch, batch_idx):
        self.schedule()

        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch
        input_image = input_image.to(self.device)
        ref_image = ref_image.to(self.device)
        output = self.video_net.forward(referframe=ref_image, input_image=input_image)

        loss = self.loss(output, input_image)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if train_args['model_type'] == 'psnr':
            quality = PSNR(output['recon_image'], input_image)
        else:
            quality = ms_ssim(output['recon_image'], input_image, data_range=1.0).item()

        return loss, quality, output["bpp_mv_y"], output["bpp_mv_z"], output["bpp_y"], output["bpp_z"], output["bpp"]


    # TODO
    def validation_step(self, batch, img_idx, output_folder):
        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch
        input_image = input_image.to(self.device)
        ref_image = ref_image.to(self.device)
        with torch.no_grad():
            output = self.video_net.forward(referframe=ref_image, input_image=input_image)
            # loss = self.loss(output, ref_image)
            loss = self.loss(output, input_image)

            # TODO 可视化
            if epoch < borders_of_steps[0]:
                self.visualization(self.current_epoch, input_image, output['x_tilde'], img_idx, output_folder)
            else:
                self.visualization(self.current_epoch, input_image, output['recon_image'], img_idx, output_folder)
            
            if train_args['model_type'] == 'psnr':
                quality = PSNR(output['recon_image'], input_image)
            else:
                quality = ms_ssim(output['recon_image'], input_image, data_range=1.0).item()

        return loss, quality, output["bpp_mv_y"], output["bpp_mv_z"], output["bpp_y"], output["bpp_z"], output["bpp"]

    def visualization(self, epoch, net_input_image, net_output_image, img_idx, output_folder):
        # 为每个权重创建一个与权重文件名相同的文件夹
        vis_folder = os.path.join(output_folder, f"model_epoch_{epoch}_visuals")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

        # 保存可视化图像
        self.save_visualization(net_input_image, net_output_image, epoch, img_idx, vis_folder)

    def save_visualization(self, input_image, output_image, epoch, img_idx, vis_folder):
        # 转换图像为可显示格式
        ref_image_np = ref_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        input_image_np = input_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        output_image_np = output_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

        # 反归一化
        ref_image_np = (ref_image_np * 255).astype(np.uint8)
        input_image_np = (input_image_np * 255).astype(np.uint8)    
        output_image_np = (output_image_np * 255).astype(np.uint8)

        # 创建图像对比图
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(ref_image_np)
        ax[0].set_title('Reference Image')
        ax[0].axis('off')
        ax[1].imshow(input_image_np)
        ax[1].set_title('Input Image')
        ax[1].axis('off')
        ax[2].imshow(output_image_np)
        ax[2].set_title('Output Image')
        ax[2].axis('off')

        # 保存图片到新建的与权重同名的文件夹
        img_save_path = os.path.join(vis_folder, f'validation_{epoch}_{img_idx}.png')
        plt.savefig(img_save_path)
        plt.close(fig)

if __name__ == "__main__":
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

    wandb.init(project="DCVC-Trainer")
    wandb.config.update(train_args)

    trainer = Trainer(train_args)
    dataset = DataSet()
    val_dataset = DataSet('H:/Data/vimeo_septuplet/vimeo_septuplet/mini_dvc_test_val.txt')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args['batch_size'], shuffle=True, num_workers=train_args['worker'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=train_args['worker'])
    for epoch in range(train_args['epochs']):
        trainer.current_epoch = epoch
        for batch_idx, (input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv) in enumerate(dataloader):
            loss, quality, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp = trainer.training_step((input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv), batch_idx)
            
            wandb.log({"loss": loss, "quality": quality})
            wandb.log({"epoch": epoch, "batch": batch_idx})
            wandb.log({"bpp_mv_y": bpp_mv_y, "bpp_mv_z": bpp_mv_z, "bpp_y": bpp_y, "bpp_z": bpp_z, "bpp": bpp})
            
        print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss}, quality({train_args['model_type']}): {quality}")
        # trainer.validation_step((input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv), batch_idx, save_folder)
        idx = 0
        losses = []
        qualities = []
        bpp_mv_ys = []
        bpp_mv_zs = []
        bpp_ys = []
        bpp_zs = []
        bpps = []
        for batch_idx, (input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv) in enumerate(val_dataloader):
            loss, quality, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp = trainer.validation_step((input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv), idx, save_folder)
            idx += 1
            losses.append(loss)
            qualities.append(quality)
            bpp_mv_ys.append(bpp_mv_y)
            bpp_mv_zs.append(bpp_mv_z)
            bpp_ys.append(bpp_y)
            bpp_zs.append(bpp_z)
            bpps.append(bpp)
        
        ave_loss = sum(losses) / len(losses)
        ave_quality = sum(qualities) / len(qualities)
        ave_bpp_mv_y = sum(bpp_mv_ys) / len(bpp_mv_ys) 
        ave_bpp_mv_z = sum(bpp_mv_zs) / len(bpp_mv_zs)
        ave_bpp_y = sum(bpp_ys) / len(bpp_ys)
        ave_bpp_z = sum(bpp_zs) / len(bpp_zs)
        ave_bpp = sum(bpps) / len(bpps)

        wandb.log({"val_loss": ave_loss, "val_quality": ave_quality, "epoch": epoch})
        wandb.log({"val_bpp_mv_y": ave_bpp_mv_y, "val_bpp_mv_z": ave_bpp_mv_z, "val_bpp_y": ave_bpp_y, "val_bpp_z": ave_bpp_z, "val_bpp": ave_bpp, "epoch": epoch})

        # save model
        torch.save(trainer.video_net.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))
        



