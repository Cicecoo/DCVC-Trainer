# 参考 https://github.com/microsoft/DCVC/issues/35
import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from src.models.DCVC_net_add_noise import DCVC_net
# from src.models.DCVC_net_full_init import DCVC_net
# from src.models.DCVC_net_quant import DCVC_net
# from src.models.DCVC_net_Spynet import DCVC_net
from src.zoo.image import model_architectures as architectures
from test_video import PSNR, ms_ssim, read_frame_to_torch
from dvc_dataset import DataSet, RawDataSet, UVGDataSet
from mmvc_dataset import VimeoDataset
# from datasetVimeo import Vimeo90kDataset

import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from utils import load_submodule_params, freeze_submodule, unfreeze_submodule, get_save_folder, clip_gradient
import random

# set CUDA_VISIBLE_DEVICES=
train_dataset_path = '/mnt/data3/zhaojunzhang/vimeo_septuplet/test.txt' # mini_dvc_test_val_1k.txt' # 
tag = 'test' if train_dataset_path.__contains__('mini') else 'main'

train_args = {
    'project': "DCVC-Trainer_remote",
    'describe': f"[25.1.8] [{tag}] old：ME 部分已经比较好，问题还在 bpp_z；排除初始化的问题；添加余弦退火；new：再次训练",
    'i_frame_model_name': "cheng2020-anchor",
    'i_frame_model_path': ["checkpoints/cheng2020-anchor-3-e49be189.pth.tar",
                           "checkpoints/cheng2020-anchor-4-98b0b468.pth.tar",
                           "checkpoints/cheng2020-anchor-5-23852949.pth.tar",
                           "checkpoints/cheng2020-anchor-6-4c052b1a.pth.tar"],
    'i_frame_model_index': 0,
    'dcvc_model_path': "checkpoints/model_dcvc_quality_0_psnr.pth",
    'test_dataset_config': "dataset_config.json",
    'worker': 12,
    'cuda': True,
    'cuda_device': 3,
    'model_type': "psnr",
    'resume': False,
    "batch_size": 4,
    "metric": "MSE", # 最小化 MSE 来最大化 PSNR
    "quality": 3,   # in [3、4、5、6]
    "gop": 10,
    "epochs": 30,
    "seed": 428571,
    "border_of_steps": [1, 4, 7, 13, 19], # [1, 4, 7, 10, 16],
    "lr_set": {
        "me1": 1e-4,
        "me2": 1e-4,
        "reconstruction": 1e-4,
        "contextual_coding": 1e-4,
        # "contextual_coding2": 1e-4,
        "all": 1e-4,
        "fine_tuning": 1e-5
        },
    "warmup_border": None,
    "decay_border": None,
    "decay_rate": None,
    "train_dataset_path": train_dataset_path,
    "loss_settings": {
        "me1": {
            "D-item": "x_tilde_dist",
            "R-item": []
        },
        "me2": {
            "D-item": "x_tilde_dist",
            "R-item": ["mv_latent_rate", "mv_prior_rate"]
        },
        "reconstruction": {
            "D-item": "x_hat_dist",
            "R-item": []
        },
        # "contextual_coding1": {
        #     "D-item": "x_hat_dist",
        #     "R-item": ["frame_latent_rate"]
        # },
        "contextual_coding": {
            "D-item": "x_hat_dist",
            "R-item": ["frame_latent_rate", "frame_prior_rate"]
        },
        "all": {
            "D-item": "x_hat_dist",
            "R-item": ["mv_latent_rate", "mv_prior_rate", "frame_latent_rate", "frame_prior_rate"]
        },
        "fine_tuning": {
            "D-item": "x_hat_dist",
            "R-item": ["mv_latent_rate", "mv_prior_rate", "frame_latent_rate", "frame_prior_rate"]
        }
    }
}

# 1.mv warmup; 2.train excluding mv; 3.train excluding mv with bit cost; 4.train all
borders_of_steps = train_args["border_of_steps"]
lr_set = train_args["lr_set"]
if train_args["decay_border"] is not None:
    decay_interval = train_args["epochs"] - train_args["decay_border"]

# 此处 index 对应文中 quality index，lambda来自于文中3.4及附录
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

        # 加载 DCVC 模型
        self.video_net = DCVC_net()
        if args['resume']:
            load_checkpoint = torch.load(args['dcvc_model_path'], map_location=torch.device('cpu'))
            self.video_net.load_dict(load_checkpoint)
        else:
            # 加载光流网络
            # load_submodule_params(self.video_net.opticFlow, "checkpoints/model_dcvc_quality_0_psnr.pth", 'opticFlow')
            pass

        self.freeze_list = [self.video_net.opticFlow,
                            self.video_net.mvEncoder,
                            self.video_net.mvpriorEncoder,
                            self.video_net.mvpriorDecoder,
                            self.video_net.auto_regressive_mv,
                            self.video_net.entropy_parameters_mv,
                            self.video_net.mvDecoder_part1,
                            self.video_net.mvDecoder_part2,
                            self.video_net.bitEstimator_z_mv
                            ]
        
        self.load_list = [
                        (self.video_net.mvEncoder, "mvEncoder"),
                        (self.video_net.mvpriorEncoder, "mvpriorEncoder"),
                        (self.video_net.mvpriorDecoder, "mvpriorDecoder"),
                        (self.video_net.auto_regressive_mv, "auto_regressive_mv"),
                        (self.video_net.entropy_parameters_mv, "entropy_parameters_mv"),
                        (self.video_net.mvDecoder_part1, "mvDecoder_part1"),
                        (self.video_net.mvDecoder_part2, "mvDecoder_part2"),
                        (self.video_net.bitEstimator_z_mv, "bitEstimator_z_mv")
                        ]

        # 加载到 gpu
        self.device = torch.device('cuda', args['cuda_device']) if args['cuda'] else torch.device('cpu')
        self.i_frame_net.to(self.device)
        self.i_frame_net.eval()
        self.video_net.to(self.device)

        # 优化器
        self.lr = lr_set
        self.optimizer = optim.AdamW(self.video_net.parameters(), lr=1e-4)

        # 超参数
        self.metric = args['metric']
        self.quality_index = args['quality']
        self.gop = args['gop']

        # 初始化
        self.current_epoch = 0
        self.step = -1
        self.step_name = None
        self.loss_items = None

    
    def schedule(self):
        update = False
        if self.current_epoch == 0:
            self.step = 1
            self.step_name = 'me1'
            update = True
            # freeze_submodule([self.video_net.opticFlow])    
        elif self.current_epoch == borders_of_steps[0]:
            self.step = 1
            self.step_name = "me2"
            update = True
        elif self.current_epoch == borders_of_steps[1]:
            self.step = 2
            self.step_name = "reconstruction"
            update = True
            freeze_submodule(self.freeze_list)
        elif self.current_epoch == borders_of_steps[2]:
            self.step = 3
            self.step_name = "contextual_coding"
            update = True
        # elif self.current_epoch == borders_of_steps[3]:
        #     self.step = 3
        #     self.step_name = "contextual_coding2"
        elif self.current_epoch == borders_of_steps[3]:
            self.step = 4
            self.step_name = "all"
            update = True
            unfreeze_submodule(self.freeze_list)
        elif self.current_epoch == borders_of_steps[4]:
            self.step = 5
            self.step_name = "fine_tuning"            
            update = True

        # 学习率调整
        base_lr = self.lr[self.step_name]
        current_lr = base_lr
        # if self.current_epoch < train_args["warmup_border"]:
        #     current_lr = base_lr * (self.current_epoch + 1) / train_args["warmup_border"]
        # elif self.step_name.startswith('reconstruction') and self.current_epoch - borders_of_steps[1] < 4:
        #     current_lr = base_lr * (self.current_epoch - borders_of_steps[1] + 1) / 4
        # elif self.step_name.startswith('contextual_coding')  and self.current_epoch - borders_of_steps[3] < 4:
        #     current_lr = base_lr * (self.current_epoch - borders_of_steps[3] + 1) / 4
        # elif self.current_epoch >= train_args["decay_border"]:
        #     current_lr = base_lr * train_args["decay_rate"] ** (self.current_epoch // decay_interval)

        if update:
            self.optimizer = optim.AdamW(filter(lambda p : p.requires_grad, self.video_net.parameters()), lr=current_lr)

        if self.step_name == "contextual_coding":
            if update:
                print(f"init scheduler: {self.step_name}, iters = {len(dataloader)} * {(borders_of_steps[2] - borders_of_steps[1])}")
                # iters = len(dataloader) * (borders_of_steps[3] - borders_of_steps[2]) / 2
                iters = (borders_of_steps[3] - borders_of_steps[2]) / 2
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iters, eta_min=0)
            else:
                pass
        else:
            self.scheduler = None

        self.loss_items = train_args["loss_settings"][self.step_name]

        print(f"step: {self.step}, step_name: {self.step_name}, update: {update}, current_lr: {current_lr}")
        print(f"loss_items: {self.loss_items}")
            
    """ 
    loss components:
        x_tilde_dist / x_hat_dist: D,
        mv_latent_rate: R(gt),
        mv_prior_rate: R(st),
        frame_latent_rate: R(yt),
        frame_prior_rate: R(zt)
    """
    loss_setting2output_obj = { # 最小化bpp来最大化压缩率
        "mv_latent_rate": "bpp_mv_y",
        "mv_prior_rate": "bpp_mv_z",
        "frame_latent_rate": "bpp_y",
        "frame_prior_rate": "bpp_z"
    }

    def loss(self, net_output, target):
        # 失真
        # warmup 时需要只获取运动补偿输出
        if self.loss_items["D-item"] == "x_tilde_dist":
            D_item = F.mse_loss(net_output["x_tilde"], target)
        else:
            D_item = F.mse_loss(net_output["recon_image"], target) 

        # 率
        R_item = 0
        for item in self.loss_items["R-item"]:
            R_item += net_output[self.loss_setting2output_obj[item]]

        # print("lambda", lambda_set[self.metric][self.quality_index])
        # print("D_item", D_item)
        # print("R_item", R_item)
        # print("bpp_mv_y", net_output["bpp_mv_y"])
        # print("bpp_mv_z", net_output["bpp_mv_z"])
        # print("bpp_y", net_output["bpp_y"])
        # print("bpp_z", net_output["bpp_z"])

        loss = lambda_set[self.metric][self.quality_index] * D_item + R_item
        return loss

    def training_step(self, batch, batch_idx):
        self.video_net.train()
        # input_image, ref_image = batch
        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv, quant_noise_z_mv = (x.to(self.device) for x in batch)
        
        ref_image = ref_image.to(self.device)
        input_image = input_image.to(self.device)
    
        # 和推理时一样将参考帧压缩
        with torch.no_grad():
            output_i = self.i_frame_net(ref_image)
            ref_image = output_i['x_hat']

        output_p = self.video_net.forward(
            referframe=ref_image, input_image=input_image, 
            # quant_noise_feature=quant_noise_feature, quant_noise_z=quant_noise_z, 
            # quant_noise_mv=quant_noise_mv, quant_noise_z_mv=quant_noise_z_mv
            )

        loss = self.loss(output_p, input_image)

        self.optimizer.zero_grad()
        loss.backward()
        # TODO  https://github.com/DeepMC-DCVC/DCVC/issues/8 必要吗？
        clip_gradient(self.optimizer, 0.5)
        self.optimizer.step()

        # if self.step > 1:
        if train_args['model_type'] == 'psnr':
            quality = PSNR(output_p['recon_image'], input_image)
        else:
            quality = ms_ssim(output_p['recon_image'], input_image, data_range=1.0).item()

        return loss, quality, output_p["bpp_mv_y"], output_p["bpp_mv_z"], output_p["bpp_y"], output_p["bpp_z"], output_p["bpp"]

    def validation_step(self, batch, img_idx, output_folder):
        self.video_net.eval()

        input_image, ref_image = batch
        
        ref_image = ref_image.to(self.device)
        input_image = input_image.to(self.device)

        output_i = self.i_frame_net(ref_image)
        ref_image = output_i['x_hat']

        with torch.no_grad():
            output = self.video_net.forward(referframe=ref_image, input_image=input_image)
            loss = self.loss(output, input_image)

            # 可视化
            if img_idx < 15:
                self.visualization(self.current_epoch, ref_image, input_image, output, img_idx, output_folder)
            
            if train_args['model_type'] == 'psnr':
                quality = PSNR(output['recon_image'], input_image)
            else:
                quality = ms_ssim(output['recon_image'], input_image, data_range=1.0).item()
                
        return loss, quality, output["bpp_mv_y"], output["bpp_mv_z"], output["bpp_y"], output["bpp_z"], output["bpp"]

    def visualization(self, epoch, net_ref_image, net_input_image, net_output, img_idx, output_folder):
        # 为每个权重创建一个与权重文件名相同的文件夹
        vis_folder = os.path.join(output_folder, f"model_epoch_{epoch}_visuals")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

        # 转换图像为可显示格式
        ref_image_np = net_ref_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        input_image_np = net_input_image[0].cpu().permute(1, 2, 0).numpy()
        warped_image_np = net_output['x_tilde'][0].cpu().permute(1, 2, 0).numpy()
        output_image_np = net_output['recon_image'][0].cpu().permute(1, 2, 0).numpy()

        # 反归一化
        ref_image_np = (ref_image_np * 255).astype(np.uint8)
        input_image_np = (input_image_np * 255).astype(np.uint8)   
        warped_image_np = (warped_image_np * 255).astype(np.uint8) 
        output_image_np = (output_image_np * 255).astype(np.uint8)

        # 创建图像对比图
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(ref_image_np)
        ax[0].set_title('Reference Image')
        ax[0].axis('off')
        ax[1].imshow(input_image_np)
        ax[1].set_title('Input Image')
        ax[1].axis('off')
        ax[2].imshow(warped_image_np)
        ax[2].set_title('Warped Image')
        ax[2].axis('off')
        ax[3].imshow(output_image_np)
        ax[3].set_title('Reconstructed Image')
        ax[3].axis('off')

        # 保存图片到新建的与权重同名的文件夹
        img_save_path = os.path.join(vis_folder, f'validation_{epoch}_{img_idx}.png')
        plt.savefig(img_save_path)
        plt.close(fig)

if __name__ == "__main__": 
    wandb.init(project=train_args["project"])
    wandb.config.update(train_args)

    if train_args["seed"] is not None:
        torch.manual_seed(train_args["seed"])
        random.seed(train_args["seed"])

    save_folder = get_save_folder()

    print("save_folder", save_folder)

    trainer = Trainer(train_args)
    # dataset = VimeoDataset(video_dir=video_dir, text_split=train_args["train_dataset"])
    # dataset = Vimeo90kDataset(data_file=train_args["train_dataset"])
    dataset = DataSet(train_dataset_path)
    # val_dataset = RawDataSet(val_dataset_path)
    # 使用UVG
    val_dataset = UVGDataSet(testfull=True)
    # val_dataset = VimeoDataset(video_dir=video_dir, text_split=train_args["test_dataset"], test=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args['batch_size'], shuffle=True, num_workers=train_args['worker'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=train_args['worker'])

    for epoch in range(train_args['epochs']):
        # 训练
        trainer.current_epoch = epoch
        with torch.no_grad():
            trainer.schedule()

        print(f"Epoch {epoch}, {trainer.step_name}, lr: {trainer.optimizer.param_groups[0]['lr']}")
        # cnt = 0
        # t_losses = 0
        # t_qualities = 0
        # t_bpps = 0
        for batch_idx, batch in enumerate(dataloader):
            loss, quality, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp = trainer.training_step(batch, batch_idx)
            # cnt += 1
            # t_losses += loss
            # t_qualities += quality
            # t_bpps += bpp

            wandb.log({"epoch": epoch, "batch": batch_idx})
            wandb.log({"loss": loss, "quality": quality, "bpp": bpp})
            wandb.log({"bpp_mv_y": bpp_mv_y, "bpp_mv_z": bpp_mv_z, "bpp_y": bpp_y, "bpp_z": bpp_z})
            group = "step" + str(trainer.step)
            wandb.log({f"{group}_epoch": epoch, f"{group}_batch": batch_idx})
            wandb.log({f"{group}_loss": loss, f"{group}_quality": quality, f"{group}_bpp": bpp})
            wandb.log({f"{group}_bpp_mv_y": bpp_mv_y, f"{group}_bpp_mv_z": bpp_mv_z, f"{group}_bpp_y": bpp_y, f"{group}_bpp_z": bpp_z})
            
        
        # print(f"Epoch {epoch} {trainer.step_name}, batch {batch_idx}, loss: {t_losses / cnt}, quality({train_args['model_type']}): {t_qualities / cnt}, bpp: {t_bpps / cnt}")

        # save model
        torch.save(trainer.video_net.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

        if trainer.scheduler is not None:
            trainer.scheduler.step()


        # 验证
        idx = 0
        losses = []
        qualities = []
        bpp_mv_ys = []
        bpp_mv_zs = []
        bpp_ys = []
        bpp_zs = []
        bpps = []
        for batch_idx, (input_image, ref_image) in enumerate(val_dataloader):
            loss, quality, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp = trainer.validation_step((input_image, ref_image), idx, save_folder)
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

        wandb.log({"epoch": epoch, "val_loss": ave_loss, "val_quality": ave_quality, "val_bpp": ave_bpp})
        wandb.log({"val_bpp_mv_y": ave_bpp_mv_y, "val_bpp_mv_z": ave_bpp_mv_z, "val_bpp_y": ave_bpp_y, "val_bpp_z": ave_bpp_z})
        print(f"Validation, epoch {epoch}, loss: {ave_loss}, quality({train_args['model_type']}): {ave_quality}, bpp: {ave_bpp}")
        print(f"bpp_mv_y: {ave_bpp_mv_y}, bpp_mv_z: {ave_bpp_mv_z}, bpp_y: {ave_bpp_y}, bpp_z: {ave_bpp_z}")
        group = "step" + str(trainer.step)
        wandb.log({"epoch": epoch, f"{group}_val_loss": ave_loss, f"{group}_val_quality": ave_quality, f"{group}_val_bpp": ave_bpp})
        wandb.log({f"{group}_val_bpp_mv_y": ave_bpp_mv_y, f"{group}_val_bpp_mv_z": ave_bpp_mv_z, f"{group}_val_bpp_y": ave_bpp_y, f"{group}_val_bpp_z": ave_bpp_z})
