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
    # 'write_stream': False,
    # 'write_recon_frame': False,
    # 'recon_bin_path': "recon_bin_path",
    # 'output_json_result_path': "required_value",  # Replace with actual required value
    'model_type': "psnr",
    'resume': False,
    "batch_size": 4,
    "metric": "MSE",
    "quality": 3,
    "gop": 10,
}

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

        device = torch.device("cuda" if args['cuda'] else "cpu")
        self.i_frame_net.to(device)
        self.i_frame_net.eval()
        self.video_net.to(device)

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

        if self.current_epoch == borders_of_steps[3]:
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
        # "x_tilde_dist": "x_tilde_dist",
        # "x_hat_dist": "x_hat_dist",
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
        # config = progressive_training_schedule(self.current_epoch)
        # lr = config["lr"]
        # nframes = config["nframes"]
        # objective = config["loss"]
        # use_avg_loss = config["avg_loss"]
        # mode = config["mode"]
        self.schedule()
        
        # self.use_weighted_loss = True if nframes >= 5 else False
        
        # q = random.randint(0, 3) if not self.single else self.quality_index

        # # Set Optimizers
        # opt = self.optimizers()
        # opt._optimizer.param_groups[0]["lr"] = lr

        # Batch: [B, T, C, H, W]
        # seq_len = batch.shape[1]
        # frames = [image.squeeze(1) for image in batch.chunk(seq_len, 1)][:nframes]

        # # I frame compression
        # with torch.no_grad():
        #     x_hat
        #     x_hat = self.i_frame_model(frames[0], q_in_ckpt=True, q_index=q)["x_hat"]
        #     dpb = {
        #         "ref_frame": x_hat,
        #         "ref_feature": None,
        #         "ref_mv_feature": None,
        #         "ref_y": None,
        #         "ref_mv_y": None,
        #     }

        # # Iterative Update
        # if mode == "inter":
        #     step = self.p_frame_model.forward_inter
        # elif mode == "recon":
        #     step = self.p_frame_model.forward_recon
        # elif mode == "all":
        #     step = self.p_frame_model.forward_all
        # else:
        #     raise NotImplementedError

        # total_psnr = AverageMeter()
        # total_bpp = AverageMeter()
        # total_mse = AverageMeter()
        # total_loss = AverageMeter()

        avg_loss = 0
        for i in range(nframes - 1):
            # (x, dpb, q_index, frame_idx):
            out_net = step(frames[i + 1], dpb, q_index=q, frame_idx=i)
            dpb = out_net["dpb"]
            
            out_criterion = self.loss(
                out_net,
                frames[i + 1],
                q_index=q,
                objective=objective,
                frame_idx=i,
            )

            if not use_avg_loss:
                opt.zero_grad()
                self.manual_backward(out_criterion["loss"])
                self.clip_gradients(
                    opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                )
                opt.step()
                # All the information in dpb are freed
                if nframes >= 3:
                    for k in dpb.keys():
                        dpb[k] = dpb[k].detach()
                        
        ref_frame = None
        frame_types = [] # 看过 h264 后理解，用于记录帧类型（I、P），给出解码顺序
        qualitys = [] # PSNR 或 MS-SSIM，给出每帧的压缩？质量
        bits = [] # 对于 I 帧，记录 bit.item()；对于 P 帧，bpp.item()*frame_pixel_num。即每帧的比特数
        bits_mv_y = [] # mv应该是运动向量，y 是？
        bits_mv_z = []
        bits_y = [] # y、z 是？
        bits_z = []

        # gop_size = args_dict['gop'] # 固定的 GOP 大小？
        frame_pixel_num = 0 
        frame_num = args_dict['frame_num'] # 在 main 计算得到

        for frame_idx in range(frame_num):
            # 读取一帧
            ori_frame = read_frame_to_torch(
                os.path.join(args_dict['dataset_path'],
                             sub_dir_name,
                             f"im{str(frame_idx+1).zfill(padding)}.png"))
            ori_frame = ori_frame.to(self.device) # 保证设备一致

            if frame_pixel_num == 0: # frame_pixel_num 只初始化一次
                frame_pixel_num = ori_frame.shape[2]*ori_frame.shape[3] # shape 为 [batch, channel, height, width]， 见 read_frame_to_torch 
            else: # 每帧 比较前后帧尺寸（保证一致）
                assert(frame_pixel_num == ori_frame.shape[2]*ori_frame.shape[3])

            if frame_idx % self.gop == 0: # I 帧
                with torch.no_grad():
                    result = self.i_frame_net(ori_frame)
                # bit = sum((torch.log(likelihoods).sum() / (-math.log(2)))  # bit = Sigma(Sigma(log(likelihoods))) / (-log(2)) = - Sigma Sigma(log2(likelihoods))
                #           for likelihoods in result["likelihoods"].values()) # likelihoods 是什么样子（为什么是二维）？
                ref_frame = result["x_hat"] # 为 P 帧初始化参考帧
                    
                # frame_types.append(0)
                # bits.append(bit.item())
                # bits_mv_y.append(0)
                # bits_mv_z.append(0)
                # bits_y.append(0)
                # bits_z.append(0)
            else: # P 帧
                result = self.video_net(ref_frame, ori_frame)
                ref_frame = result['recon_image'] # 由于没有B帧，下一帧如果需要参考帧，一定是上一帧，对于 P 帧是重建后的帧？

                loss = self.loss(result, ori_frame)

                # TODO: froze 部分网络
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                 
                # bpp = result['bpp']
                # frame_types.append(1)
                # bits.append(bpp.item()*frame_pixel_num)
                # bits_mv_y.append(result['bpp_mv_y'].item()*frame_pixel_num)
                # bits_mv_z.append(result['bpp_mv_z'].item()*frame_pixel_num)
                # bits_y.append(result['bpp_y'].item()*frame_pixel_num)
                # bits_z.append(result['bpp_z'].item()*frame_pixel_num)

            ref_frame = ref_frame.clamp_(0, 1) # clamp 限幅

            if train_args['model_type'] == 'psnr':
                qualitys.append(PSNR(ref_frame, ori_frame))
            else:
                qualitys.append(ms_ssim(ref_frame, ori_frame, data_range=1.0).item())
 
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pass
            # nframes = 5
            # objective = ["mv_rate", "x_rate", "x_dist"]
            # self.use_weighted_loss = True if nframes >= 5 else False

            # for q in range(4):
            #     # Set Optimizers
            #     # Batch: [B, T, C, H, W]
            #     seq_len = batch.shape[1]
            #     frames = [image.squeeze(1) for image in batch.chunk(seq_len, 1)]
            #     recon_frames = []

            #     # I frame compression
            #     # (x, q_in_ckpt=False, q_index=None):
            #     x_hat = self.i_frame_model(frames[0], q_in_ckpt=True, q_index=q)[
            #         "x_hat"
            #     ]
            #     dpb = {
            #         "ref_frame": x_hat,
            #         "ref_feature": None,
            #         "ref_mv_feature": None,
            #         "ref_y": None,
            #         "ref_mv_y": None,
            #     }
            #     recon_frames.append(x_hat)

            #     # Iterative Update
            #     step = self.p_frame_model.forward_all

            #     total_psnr = AverageMeter()
            #     total_bpp = AverageMeter()
            #     total_mse = AverageMeter()
            #     total_loss = AverageMeter()

            #     for i in range(nframes - 1):
            #         # (x, dpb, q_index, frame_idx):
            #         out_net = step(frames[i + 1], dpb, q_index=q, frame_idx=(i))
            #         out_criterion = self.loss(
            #             out_net,
            #             frames[i + 1],
            #             q_index=q,
            #             objective=objective,
            #             frame_idx=i,
            #         )

            #         dpb = out_net["dpb"]
            #         recon_frames.append(dpb["ref_frame"])

            #         total_psnr.update(out_criterion["psnr"].item())
            #         total_bpp.update(out_criterion["bpp"].item())
            #         total_mse.update(out_criterion["mse"].item())
            #         total_loss.update(out_criterion["loss"].item())

            #     self.log_dict(
            #         {
            #             f"val_avg_psnr/q{q}": total_psnr.avg,
            #             f"val_avg_bpp/q{q}": total_bpp.avg,
            #             f"val_avg_mse/q{q}": total_mse.avg,
            #             f"val_avg_loss/q{q}": total_loss.avg,
            #         },
            #         sync_dist=True,
            #     )

            #     if batch_idx == 2:
            #         self.log_images(
            #             {
            #                 f"val_x_ori/q{q}": torch.cat(frames, dim=0),
            #                 f"val_x_recon/q{q}": torch.cat(recon_frames, dim=0),
            #             },
            #             batch_idx
            #         )

    # def configure_optimizers(self):
    #     parameters = {n for n, p in self.p_frame_model.named_parameters()}
    #     params_dict = dict(self.p_frame_model.named_parameters())

    #     optimizer = optim.AdamW(
    #         (params_dict[n] for n in sorted(parameters)),
    #         lr=1e-4,  # default
    #     )
    #     # optimizer = optim.Adam(
    #     #     (params_dict[n] for n in sorted(parameters)),
    #     #     lr=1e-4,  # default
    #     # )

    #     return {
    #         "optimizer": optimizer,
    #     }

