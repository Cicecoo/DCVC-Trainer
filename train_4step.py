# 参考 https://github.com/microsoft/DCVC/issues/35
import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
# from src.models.DCVC_net_add_noise import DCVC_net
from src.models.IBVC_DCVC_net import DCVC_net
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


train_dataset_path = '/mnt/data3/zhaojunzhang/vimeo_septuplet/test.txt' # mini_dvc_test_val_1k.txt'

train_args = {
    'project': "DCVC-Trainer_remote",
    'describe': "[25.1.2] epoch 7 bbp_z 骤降问题，不添加学习率调整，每轮重置优化器",
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
    'cuda_device': 2,
    'model_type': "psnr",
    'resume': False,
    "batch_size": 4,
    "metric": "MSE", # 最小化 MSE 来最大化 PSNR
    "quality": 3,   # in [3、4、5、6]
    "gop": 10,
    "epochs": 26,
    "seed": 19,
    "border_of_steps": [4, 7, 10, 16], #[1, 4, 7, 10, 16],
    "lr_set": {
        # "me1": 1e-4,
        "me": 1e-4,
        "reconstruction": 1e-4,
        "contextual_coding": 1e-4,
        "all": 1e-4,
        "fine_tuning": 1e-5
        },
    "warmup_border": None,
    "decay_border": None,
    "decay_rate": None,
    # "train_dataset": '/mnt/data3/zhaojunzhang/vimeo_septuplet/sep_trainlist.txt',
    # "test_dataset": '/mnt/data3/zhaojunzhang/vimeo_septuplet/sep_testlist.txt',
    "train_dataset_path": train_dataset_path,
}

# video_dir = '/mnt/data3/zhaojunzhang/vimeo_septuplet/sequences'

# 1.mv warmup; 2.train excluding mv; 3.train excluding mv with bit cost; 4.train all
borders_of_steps = train_args["border_of_steps"]
lr_set = train_args["lr_set"]
decay_interval = train_args["epochs"] - train_args["decay_border"]

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
                            # self.video_net.mvpriorEncoder,
                            # self.video_net.mvpriorDecoder, # 是否需要?
                            # self.video_net.auto_regressive_mv,
                            # self.video_net.entropy_parameters_mv,
                            self.video_net.mvDecoder_part1,
                            self.video_net.mvDecoder_part2,
                            # self.video_net.bitEstimator_z_mv
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

        # 加载ME
        # load_submodule_params(self.video_net.opticFlow, "checkpoints/model_dcvc_quality_0_psnr.pth", 'opticFlow')
        # whole_module_state_dict = torch.load("checkpoints/model_dcvc_quality_0_psnr.pth")
        # for module in self.load_list:
        #     load_submodule_params_(module[0], whole_module_state_dict, module[1])

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
        self.loss_settings = dict()

        # self.freeze_list1 = [self.video_net.bitEstimator_z,
        #                     self.video_net.feature_extract,
        #                     self.video_net.context_refine,
        #                     # self.video_net.gaussian_encoder,
        #                     self.video_net.contextualEncoder,
        #                     self.video_net.contextualDecoder_part1,
        #                     self.video_net.contextualDecoder_part2, 
        #                     self.video_net.priorEncoder,
        #                     self.video_net.priorDecoder,
        #                     self.video_net.entropy_parameters, 
        #                     self.video_net.auto_regressive,
        #                     self.video_net.temporalPriorEncoder
        #                     ]

        self.step = 0
        self.step_name = None

        
    
    def schedule(self):
        if self.current_epoch == 0:
            self.step = 1
            self.step_name = 'me'
            # freeze_submodule([self.video_net.opticFlow])  
        # elif self.current_epoch == borders_of_steps[0]:
        #     self.step = 2
        #     self.step_name = "me2"      
        #     update = True       
        #     pass
        elif self.current_epoch == borders_of_steps[0]:
            self.step = 2
            self.step_name = "reconstruction"
            freeze_submodule(self.freeze_list)        
        elif self.current_epoch == borders_of_steps[1]:
            self.step = 3
            self.step_name = "contextual_coding"
            # 根据 https://github.com/DeepMC-DCVC/DCVC/issues/8 "the whole optical motion estimation, MV encoding and decoding parts are fixed during this step"            
        elif self.current_epoch == borders_of_steps[2]:
            self.step = 4
            self.step_name = "all"
            unfreeze_submodule(self.freeze_list)
        elif self.current_epoch == borders_of_steps[3]:
            self.step = 5
            self.step_name = "fine_tuning"

        # 学习率调整
        # base_lr = self.lr[self.step_name]
        # current_lr = base_lr
        # if self.current_epoch < train_args["warmup_border"]:
        #     current_lr = base_lr * (self.current_epoch + 1) / train_args["warmup_border"]
        # elif self.step_name == 'reconstruction':
        #     current_lr = base_lr * (self.current_epoch - borders_of_steps[1] + 1) / (borders_of_steps[2] - borders_of_steps[1])
        # elif self.step_name == 'contextual_coding':
        #     current_lr = base_lr * (self.current_epoch - borders_of_steps[2] + 1) / (borders_of_steps[3] - borders_of_steps[2])
        # elif self.current_epoch >= train_args["decay_border"]:
        #     current_lr = base_lr * train_args["decay_rate"] ** (self.current_epoch // decay_interval)

        # print(f"update: {update}, current_lr: {current_lr}")

        # if update:
            # self.optimizer = optim.AdamW(filter(lambda p : p.requires_grad, self.video_net.parameters()), lr=current_lr)
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = current_lr

        current_lr = self.lr[self.step_name]
        self.optimizer = optim.AdamW(filter(lambda p : p.requires_grad, self.video_net.parameters()), lr=current_lr)

        loss_settings = dict()
        self.loss_settings.clear()
        loss_settings["step"] = self.step
        loss_settings["name"] = self.step_name

        if self.step_name == "me":
            loss_settings["D-item"] = "x_tilde_dist" 
        else:
            loss_settings["D-item"] = "x_hat_dist"
        
        loss_settings["R-item"] = []
        if self.step_name == "me" or self.step_name == "all" or self.step_name == "fine_tuning":
            loss_settings["R-item"].append("mv_latent_rate") # gt 
            loss_settings["R-item"].append("mv_prior_rate") # st
        if self.step_name == "contextual_coding" or self.step_name == "all" or self.step_name == "fine_tuning":
            loss_settings["R-item"].append("frame_latent_rate") # yt
            loss_settings["R-item"].append("frame_prior_rate") # zt
        # 更新 trainer 的 loss_settings 
        self.loss_settings = loss_settings

        print(f"step: {self.step}, step_name: {self.step_name}, current_lr: {current_lr}")
        print(f"loss_settings: {self.loss_settings}")
        
    """
    loss components:
        x_tilde_dist / x_hat_dist: D,
        mv_latent_rate: R(gt),
        mv_prior_rate: R(st),
        frame_latent_rate: R(yt),
        frame_prior_rate: R(zt)
    """
    loss_setting2output_obj = {
        "mv_latent_rate": "bpp_mv_y",
        "mv_prior_rate": "bpp_mv_z",
        "frame_latent_rate": "bpp_y",
        "frame_prior_rate": "bpp_z"
    }

    def loss(self, net_output, target):
        # 失真
        if self.loss_settings["D-item"] == "x_tilde_dist":
            D_item = F.mse_loss(net_output["x_tilde"], target)
        else: # x_hat_dist
            D_item = F.mse_loss(net_output["recon_image"], target) 

        # 率
        R_item = 0
        for item in self.loss_settings["R-item"]:
            R_item += net_output[self.loss_setting2output_obj[item]]

        print("lambda", lambda_set[self.metric][self.quality_index])
        print("D_item", D_item)
        print("R_item", R_item)
        # print("bpp_mv_y", net_output["bpp_mv_y"])
        # print("bpp_mv_z", net_output["bpp_mv_z"])
        # print("bpp_y", net_output["bpp_y"])
        # print("bpp_z", net_output["bpp_z"])

        loss = lambda_set[self.metric][self.quality_index] * D_item + R_item
        return loss


    def training_step(self, batch, batch_idx):
        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv, quant_noise_z_mv = (x.to(self.device) for x in batch)
        
        ref_image = ref_image.to(self.device)
        input_image = input_image.to(self.device)
    
        # 和推理时一样将参考帧压缩
        with torch.no_grad():
            output_i = self.i_frame_net(ref_image)
            ref_image = output_i['x_hat']

        output_p = self.video_net.forward(
            referframe=ref_image, 
            input_image=input_image, 
            quant_noise_feature=quant_noise_feature, 
            quant_noise_z=quant_noise_z, 
            quant_noise_mv=quant_noise_mv, 
            quant_noise_z_mv=quant_noise_z_mv
            )

        loss = self.loss(output_p, input_image)

        self.optimizer.zero_grad()
        loss.backward()

        clip_gradient(self.optimizer, 0.5)
        self.optimizer.step()

        if train_args['model_type'] == 'psnr':
            quality = PSNR(output_p['recon_image'], input_image)
        else:
            quality = ms_ssim(output_p['recon_image'], input_image, data_range=1.0).item()

        return loss, quality, output_p["bpp_mv_y"], output_p["bpp_mv_z"], output_p["bpp_y"], output_p["bpp_z"], output_p["bpp"]


    def validation_step(self, batch, img_idx, output_folder):
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
        input_image_np = net_input_image[0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        warped_image_np = net_output['x_tilde'][0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        output_image_np = net_output['recon_image'][0].cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

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
        trainer.video_net.train()
        trainer.current_epoch = epoch
        with torch.no_grad():
            trainer.schedule() 
        # for batch_idx, (input_image, ref_image) in enumerate(dataloader):
        for batch_idx, batch in enumerate(dataloader):
            # print(f"Epoch {epoch}, batch {batch_idx}")
            # print(input_image.shape, ref_image.shape)
            # loss, quality, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp = trainer.training_step((input_image, ref_image), batch_idx)
            loss, quality, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp = trainer.training_step(batch, batch_idx)
            
            wandb.log({"loss": loss, "quality": quality})
            wandb.log({"epoch": epoch, "batch": batch_idx})
            wandb.log({"bpp_mv_y": bpp_mv_y, "bpp_mv_z": bpp_mv_z, "bpp_y": bpp_y, "bpp_z": bpp_z, "bpp": bpp})
            group = "step" + str(trainer.step)
            wandb.log({f"{group}_loss": loss, f"{group}_quality": quality})
            wandb.log({f"{group}_bpp_mv_y": bpp_mv_y, f"{group}_bpp_mv_z": bpp_mv_z, f"{group}_bpp_y": bpp_y, f"{group}_bpp_z": bpp_z, f"{group}_bpp": bpp})
            wandb.log({f"{group}_epoch": epoch, f"{group}_batch": batch_idx})
            
        print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss}, quality({train_args['model_type']}): {quality}")

        # save model
        torch.save(trainer.video_net.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

        # 验证
        trainer.video_net.eval()
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

        wandb.log({"val_loss": ave_loss, "val_quality": ave_quality, "epoch": epoch})
        wandb.log({"val_bpp_mv_y": ave_bpp_mv_y, "val_bpp_mv_z": ave_bpp_mv_z, "val_bpp_y": ave_bpp_y, "val_bpp_z": ave_bpp_z, "val_bpp": ave_bpp, "epoch": epoch})
        print(f"Epoch {epoch}, val_loss: {ave_loss}, val_quality({train_args['model_type']}): {ave_quality}")
        print(f"Epoch {epoch}, val_bpp_mv_y: {ave_bpp_mv_y}, val_bpp_mv_z: {ave_bpp_mv_z}, val_bpp_y: {ave_bpp_y}, val_bpp_z: {ave_bpp_z}, val_bpp: {ave_bpp}")

        group = "step" + str(trainer.step)
        wandb.log({f"{group}_val_loss": ave_loss, f"{group}_val_quality": ave_quality, "epoch": epoch})
        wandb.log({f"{group}_val_bpp_mv_y": ave_bpp_mv_y, f"{group}_val_bpp_mv_z": ave_bpp_mv_z, f"{group}_val_bpp_y": ave_bpp_y, f"{group}_val_bpp_z": ave_bpp_z, f"{group}_val_bpp": ave_bpp, "epoch": epoch})
