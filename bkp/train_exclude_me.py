# 参考 https://github.com/microsoft/DCVC/issues/35
import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
# from src.models.DCVC_net import DCVC_net
from src.zoo.image import model_architectures as architectures
from test_video import PSNR, ms_ssim, read_frame_to_torch
from dvc_dataset import DataSet, RawDataSet

import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from utils import get_save_folder, clip_gradient
import random

from src.models.DCVC_net_me_excluded import DCVC_net


train_dataset_path = 'H:/Data/vimeo_septuplet/vimeo_septuplet/huge_dvc_test.txt'
val_dataset_path = "H:/Data/vimeo_septuplet/vimeo_septuplet/huge_dvc_test_val.txt"

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
    "quality": 6,   # 3、4、5、6
    "gop": 10,
    "epochs": 16,
    "seed": 0,
    "note": "me excluded"
}

# 1.mv warmup; 2.train excluding mv; 3.train excluding mv with bit cost; 4.train all
borders_of_steps = [1, 4, 10] # 参考 https://arxiv.org/pdf/2111.13850v1 "single" stage

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
            # args['i_frame_model_path'][args['i_frame_model_index']], 
            "checkpoints/cheng2020-anchor-3-e49be189.pth.tar",
            map_location=torch.device('cpu')
            )
        self.i_frame_net = architectures[args['i_frame_model_name']].from_state_dict(i_frame_load_checkpoint)
        for param in self.i_frame_net.parameters():
            param.requires_grad = False

        self.video_net = DCVC_net()

        # 加载到 gpu
        self.device = torch.device("cuda" if args['cuda'] else "cpu")
        self.i_frame_net.to(self.device)
        self.i_frame_net.eval()
        self.video_net.to(self.device)

        # 优化器
        self.lr = [1e-4] 
        self.optimizer = optim.AdamW(self.video_net.parameters(), lr=self.lr[0])

        # 超参数
        self.metric = args['metric']
        self.quality_index = args['quality']
        self.gop = args['gop']

        # 初始化
        self.current_epoch = 0

    def loss(self, net_output, target):
        D_item = F.mse_loss(net_output["recon_image"], target) 

        R_item = net_output["bpp_y"] + net_output["bpp_z"]

        loss = lambda_set[self.metric][self.quality_index] * D_item + R_item
        return loss

    def training_step(self, batch, batch_idx):
        input_image, ref_image = batch
        
        ref_image = ref_image.to(self.device)
        input_image = input_image.to(self.device)
        
        with torch.no_grad():
            output_i = self.i_frame_net(ref_image)
            ref_image = output_i['x_hat']

        output_p = self.video_net.forward_exclude_me(referframe=ref_image, input_image=input_image)

        loss = self.loss(output_p, input_image)

        self.optimizer.zero_grad()
        loss.backward()
        # TODO  https://github.com/DeepMC-DCVC/DCVC/issues/8 有clip，必要吗？
        # clip_gradient(self.optimizer, 5)
        self.optimizer.step()

        if train_args['model_type'] == 'psnr':
            quality = PSNR(output_p['recon_image'], input_image)
        else:
            quality = ms_ssim(output_p['recon_image'], input_image, data_range=1.0).item()
        
        return loss, quality, output_p["bpp_y"], output_p["bpp_z"], output_p["bpp"]

    def validation_step(self, batch, img_idx, output_folder):
        input_image, ref_image = batch
        
        ref_image = ref_image.to(self.device)
        input_image = input_image.to(self.device)

        with torch.no_grad():
            output_i = self.i_frame_net(ref_image)
            ref_image = output_i['x_hat']

            output = self.video_net.forward_exclude_me(referframe=ref_image, input_image=input_image)
            loss = self.loss(output, input_image)

            # 可视化
            if img_idx < 15:
                self.visualization(self.current_epoch, ref_image, input_image, output, img_idx, output_folder)
            
            if train_args['model_type'] == 'psnr':
                quality = PSNR(output['recon_image'], input_image)
            else:
                quality = ms_ssim(output['recon_image'], input_image, data_range=1.0).item()
                
        return loss, quality, output["bpp_y"], output["bpp_z"], output["bpp"]

    def visualization(self, epoch, net_ref_image, net_input_image, net_output, img_idx, output_folder):
        # 为每个权重创建一个与权重文件名相同的文件夹
        vis_folder = os.path.join(output_folder, f"model_epoch_{epoch}_visuals")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

        # 转换图像为可显示格式
        ref_image_np = net_ref_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        input_image_np = net_input_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        output_image_np = net_output['recon_image'][0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

        # 反归一化
        ref_image_np = (ref_image_np * 255).astype(np.uint8)
        input_image_np = (input_image_np * 255).astype(np.uint8)   
        output_image_np = (output_image_np * 255).astype(np.uint8)

        # 创建图像对比图
        fig, ax = plt.subplots(1, 3, figsize=(12, 4)) 
        ax[0].imshow(ref_image_np)
        ax[0].set_title('Reference Image')
        ax[0].axis('off')
        ax[1].imshow(input_image_np)
        ax[1].set_title('Input Image')
        ax[1].axis('off')
        ax[2].imshow(output_image_np)
        ax[2].set_title('Reconstructed Image')
        ax[2].axis('off')

        # 保存图片到新建的与权重同名的文件夹
        img_save_path = os.path.join(vis_folder, f'validation_{epoch}_{img_idx}.png')
        plt.savefig(img_save_path)
        plt.close(fig)

if __name__ == "__main__": 
    wandb.init(project="DCVC-Trainer")
    wandb.config.update(train_args)

    if train_args["seed"] is not None:
        torch.manual_seed(train_args["seed"])
        random.seed(train_args["seed"])

    save_folder = get_save_folder()

    trainer = Trainer(train_args)
    dataset = DataSet(train_dataset_path)
    val_dataset = RawDataSet(val_dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args['batch_size'], shuffle=True, num_workers=train_args['worker'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=train_args['worker'])

    for epoch in range(train_args['epochs']):
        # 训练
        trainer.current_epoch = epoch
        for batch_idx, (input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv) in enumerate(dataloader):
            loss, quality, bpp_y, bpp_z, bpp = trainer.training_step((input_image, ref_image), batch_idx)
            
            wandb.log({"loss": loss, "quality": quality})
            wandb.log({"epoch": epoch, "batch": batch_idx})
            wandb.log({"bpp_y": bpp_y, "bpp_z": bpp_z, "bpp": bpp})

            # group = "step" + str(trainer.step)
            # wandb.log({f"{group}_loss": loss, f"{group}_quality": quality})
            # wandb.log({f"{group}_bpp_y": bpp_y, f"{group}_bpp_z": bpp_z, f"{group}_bpp": bpp})
            # wandb.log({f"{group}_epoch": epoch, f"{group}_batch": batch_idx})
            
        print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss}, quality({train_args['model_type']}): {quality}")

        # save model
        torch.save(trainer.video_net.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

        # 验证
        idx = 0
        losses = []
        qualities = []
        bpp_ys = []
        bpp_zs = []
        bpps = []
        for batch_idx, (input_image, ref_image) in enumerate(val_dataloader):
            loss, quality, bpp_y, bpp_z, bpp = trainer.validation_step((input_image, ref_image), idx, save_folder)
            idx += 1
            losses.append(loss)
            qualities.append(quality)
            bpp_ys.append(bpp_y)
            bpp_zs.append(bpp_z)
            bpps.append(bpp)
        
        ave_loss = sum(losses) / len(losses)
        ave_quality = sum(qualities) / len(qualities)
        ave_bpp_y = sum(bpp_ys) / len(bpp_ys)
        ave_bpp_z = sum(bpp_zs) / len(bpp_zs)
        ave_bpp = sum(bpps) / len(bpps)

        wandb.log({"val_loss": ave_loss, "val_quality": ave_quality, "epoch": epoch})
        wandb.log({"val_bpp_y": ave_bpp_y, "val_bpp_z": ave_bpp_z, "val_bpp": ave_bpp, "epoch": epoch})

        # group = "step" + str(trainer.step)
        # wandb.log({f"{group}_val_loss": ave_loss, f"{group}_val_quality": ave_quality, "epoch": epoch})
        # wandb.log({f"{group}_val_bpp_y": ave_bpp_y, f"{group}_val_bpp_z": ave_bpp_z, f"{group}_val_bpp": ave_bpp, "epoch": epoch})
        
        



