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
    "metric": "MSE",
    "quality": 3,
    "gop": 10,
}

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


# 1.mv warmup; 2.train excluding mv; 3.train excluding mv with bit cost; 4.train all
borders_of_steps = [1, 4, 7]  

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
        # 模型
        i_frame_load_checkpoint = torch.load(
            args['i_frame_model_path'][args['i_frame_model_index']], 
            map_location=torch.device('cpu')
            )
        self.i_frame_net = architectures[args['i_frame_model_name']].from_state_dict(i_frame_load_checkpoint).eval()

        self.video_net = DCVC_net()
        if args['resume']:
            load_checkpoint = torch.load(args['dcvc_model_path'], map_location=torch.device('cpu'))
            self.video_net.load_dict(load_checkpoint)

        self.device = torch.device("cuda" if args['cuda'] else "cpu")
        self.i_frame_net.to(self.device)
        self.i_frame_net.eval()
        self.video_net.to(self.device)

        self.optimizer = optim.Adam(self.video_net.parameters(), lr=1e-4)

        # 超参数
        self.metric = args['metric']
        self.quality_index = args['quality']
        self.gop = args['gop']

        # 
        self.current_epoch = 0
        self.current_step = 0
        self.loss_settings = dict()
    
    def schedule(self):
        if self.current_epoch < borders_of_steps[0]:
            step = 0
            name = "me"
        elif self.current_epoch < borders_of_steps[1]:
            step = 1
            name = "reconstruction"
        elif self.current_epoch < borders_of_steps[2]:
            step = 2
            name = "contextual_coding"
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
        if step == 0:
            # x_tilde 是 warped frame in pixel domain，见附录B Step1
            # loss_settings["components"].append("x_tilde_dist") 
            pass
        else:
            # loss_settings["components"].append("x_hat_dist")
            pass

        if step == 0 or step == 3:
            loss_settings["components"].append("mv_latent_rate") # gt 
            loss_settings["components"].append("mv_prior_rate") # st
        if step == 2 or step == 3:
            loss_settings["components"].append("frame_latent_rate") # yt
            loss_settings["components"].append("frame_prior_rate") # zt
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
    loss_setting2output_obj = {
        "mv_latent_rate": "bpp_mv_z",
        "mv_prior_rate": "bpp_mv_y",
        "frame_latent_rate": "bpp_z",
        "frame_prior_rate": "bpp_y"
    }

    def loss(self, net_output, target):
        # warmup 时需要只获取运动补偿输出
        loss = lambda_set[self.metric][self.quality_index] * F.mse_loss(net_output["recon_image"], target)
        for component in self.loss_settings["components"]:
            loss += net_output[self.loss_setting2output_obj[component]]
        return loss

    def training_step(self, batch, batch_idx):
        self.schedule()

        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch
        input_image = input_image.to(self.device)
        ref_image = ref_image.to(self.device)
        output = self.video_net.forward(referframe=ref_image, input_image=input_image)

        loss = self.loss(output, ref_image)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if train_args['model_type'] == 'psnr':
            quality = PSNR(output['recon_image'], ref_image)
        else:
            quality = ms_ssim(output['recon_image'], ref_image, data_range=1.0).item()

        return loss, quality
        

    def validation_step(self, batch, batch_idx):
        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch
        with torch.no_grad():
            output = self.video_net.forward(referframe=ref_image, input_image=input_image)
            loss = self.loss(output, ref_image)
            
            if train_args['model_type'] == 'psnr':
                quality = PSNR(output['recon_image'], ref_image)
            else:
                quality = ms_ssim(output['recon_image'], ref_image, data_range=1.0).item()

        return loss, quality


if __name__ == "__main__":
    wandb.init(project="DCVC-Trainer")
    wandb.config.update(train_args)

    trainer = Trainer(train_args)
    dataset = DataSet()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args['batch_size'], shuffle=True, num_workers=train_args['worker'])
    for epoch in range(12):
        trainer.current_epoch = epoch
        for batch_idx, (input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv) in enumerate(dataloader):
            loss, quality = trainer.training_step((input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv), batch_idx)
            
            
            wandb.log({"loss": loss, "quality": quality})
            wandb.log({"epoch": epoch, "batch": batch_idx})
            
        print(f"Epoch {epoch}, batch {batch_idx}, loss: {loss}, quality({train_args['model_type']}): {quality}")

        # save model
        torch.save(trainer.video_net.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))
        

    # def training_step(self, batch, batch_idx):
    #     self.schedule()
    #     ref_frame = None
    #     qualitys = [] # PSNR 或 MS-SSIM，给出每帧的压缩？质量
    #     # bits = [] # 对于 I 帧，记录 bit.item()；对于 P 帧，bpp.item()*frame_pixel_num。即每帧的比特数
    #     # bits_mv_y = [] 
    #     # bits_mv_z = []
    #     # bits_y = [] 
    #     # bits_z = []
    #     frame_pixel_num = 0 
    #     frame_num = args_dict['frame_num']

    #     for frame_idx in range(frame_num):
    #         # 读取一帧
    #         ori_frame = read_frame_to_torch(
    #             os.path.join(args_dict['dataset_path'],
    #                          sub_dir_name,
    #                          f"im{str(frame_idx+1).zfill(padding)}.png"))
    #         ori_frame = ori_frame.to(self.device) # 保证设备一致

    #         if frame_pixel_num == 0:
    #             frame_pixel_num = ori_frame.shape[2]*ori_frame.shape[3] # shape 为 [batch, channel, height, width]， 见 read_frame_to_torch 
    #         else: # 每帧 比较前后帧尺寸（保证一致）
    #             assert(frame_pixel_num == ori_frame.shape[2]*ori_frame.shape[3])

    #         if frame_idx % self.gop == 0: # I 帧
    #             with torch.no_grad():
    #                 result = self.i_frame_net(ori_frame)
    #             ref_frame = result["x_hat"]
    #         else: # P 帧
    #             result = self.video_net(ref_frame, ori_frame)
    #             ref_frame = result['recon_image'] 

    #             loss = self.loss(result, ori_frame)

    #             # TODO: freeze 部分网络
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
                 
    #             # bpp = result['bpp']
    #             # frame_types.append(1)
    #             # bits.append(bpp.item()*frame_pixel_num)
    #             # bits_mv_y.append(result['bpp_mv_y'].item()*frame_pixel_num)
    #             # bits_mv_z.append(result['bpp_mv_z'].item()*frame_pixel_num)
    #             # bits_y.append(result['bpp_y'].item()*frame_pixel_num)
    #             # bits_z.append(result['bpp_z'].item()*frame_pixel_num)

    #         ref_frame = ref_frame.clamp_(0, 1) # clamp 限幅

    #         if train_args['model_type'] == 'psnr':
    #             qualitys.append(PSNR(ref_frame, ori_frame))
    #         else:
    #             qualitys.append(ms_ssim(ref_frame, ori_frame, data_range=1.0).item())
 

    #     with torch.no_grad():
    #         pass


