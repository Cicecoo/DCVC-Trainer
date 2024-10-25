# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1  # video_net 里是一些模块、工具
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..utils.stream_helper import get_downsampled_shape
from ..layers.layers import MaskedConv2d, subpel_conv3x3


class DCVC_net(nn.Module):
    def __init__(self):
        super().__init__()
        out_channel_mv = 128 # 运动向量的输出通道数
        out_channel_N = 64  
        out_channel_M = 96

        self.out_channel_mv = out_channel_mv
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.bitEstimator_z = BitEstimator(out_channel_N) # BitEstimator：
        self.bitEstimator_z_mv = BitEstimator(out_channel_N)

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_encoder = GaussianEncoder()

        # self.mvEncoder = nn.Sequential( # 扩展通道数、下采样 
        #     # 为什么输出通道数是 out_channel_mv？
        #     nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
        #     GDN(out_channel_mv), # 作用是什么？
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        #     GDN(out_channel_mv),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        #     GDN(out_channel_mv),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        # )

        # self.mvDecoder_part1 = nn.Sequential( # 上采样
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
        #                        stride=2, padding=1, output_padding=1), # ConvTranspose2d：反卷积，作用是上采样
        #     GDN(out_channel_mv, inverse=True),  # GDN：Generalized Divisive Normalization，广义除法归一化
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
        #                        stride=2, padding=1, output_padding=1),
        #     GDN(out_channel_mv, inverse=True),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
        #                        stride=2, padding=1, output_padding=1),
        #     GDN(out_channel_mv, inverse=True),
        #     nn.ConvTranspose2d(out_channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        # )

        # self.mvDecoder_part2 = nn.Sequential(
        #     nn.Conv2d(5, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(64, 2, 3, stride=1, padding=1),
        # )

        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3, stride=1, padding=1)
        )

        self.mvpriorEncoder = nn.Sequential( # 压缩通道数、下采样
            nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.mvpriorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N * 3 // 2, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N * 3 // 2, out_channel_mv*2, 3, stride=1, padding=1)
        )

        self.entropy_parameters = nn.Sequential( # 压缩通道数
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d( # 训练时像注意力机制一样，只能看到自己之前的信息
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )

        self.auto_regressive_mv = MaskedConv2d(
            out_channel_mv, 2 * out_channel_mv, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 12 // 3, out_channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 10 // 3, out_channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 8 // 3, out_channel_mv * 6 // 3, 1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.opticFlow = ME_Spynet()

        # http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch

    def motioncompensation(self, ref, mv): # 运动补偿 
        # 渐进训练 step1 的 x tilde, 不应该经过 feature extractor？
        # x_tilde = flow_warp(ref, mv) 

        ref_feature = self.feature_extract(ref) # 对应原文 feature extractor
        prediction_init = flow_warp(ref_feature, mv) # 此warp应该不是原文提到的warp？ 光流场变换：将参考帧的特征图根据光流场变换到当前帧
        context = self.context_refine(prediction_init)

        return context #, x_tilde

    def mv_refine(self, ref, mv):
        return self.mvDecoder_part2(torch.cat((mv, ref), 1)) + mv

    def quantize(self, inputs, mode, means=None): 
        assert(mode == "dequantize")
        outputs = inputs.clone()
        outputs -= means
        outputs = torch.round(outputs)
        outputs += means
        return outputs

    def feature_probs_based_sigma(self, feature, mean, sigma):
        outputs = self.quantize( # 为什么是 dequantize？
            feature, "dequantize", mean
        )
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)) # 用信息熵估计压缩后的比特数
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def iclr18_estrate_bits_z_mv(self, z_mv):
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - self.bitEstimator_z_mv(z_mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def update(self, force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    # 主干
    def encode_decode(self, ref_frame, input_image, output_path):
        encoded = self.encode(ref_frame, input_image, output_path)
        decoded = self.decode(ref_frame, output_path)
        encoded['recon_image'] = decoded
        return encoded

    def encode(self, ref_frame, input_image, output_path):
        from ..utils.stream_helper import encode_p
        N, C, H, W = ref_frame.size()
        compressed = self.compress(ref_frame, input_image)
        # 需要保留的信息
        mv_y_string = compressed['mv_y_string']
        mv_z_string = compressed['mv_z_string']
        y_string = compressed['y_string']
        z_string = compressed['z_string']
        encode_p(H, W, mv_y_string, mv_z_string, y_string, z_string, output_path)
        return {
            'bpp_mv_y': compressed['bpp_mv_y'],
            'bpp_mv_z': compressed['bpp_mv_z'],
            'bpp_y': compressed['bpp_y'],
            'bpp_z': compressed['bpp_z'],
            'bpp': compressed['bpp'],
        }

    def decode(self, ref_frame, input_path):
        from ..utils.stream_helper import decode_p
        height, width, mv_y_string, mv_z_string, y_string, z_string = decode_p(input_path)
        return self.decompress(ref_frame, mv_y_string, mv_z_string,
                               y_string, z_string, height, width)

    def compress_ar(self, y, kernel_size, context_prediction, params, entropy_parameters): # ar: auto-regressive
        # params 由 temporalPriorEncoder 的输出和 priorDecoder 的输出拼接而成
        kernel_size = 5
        padding = (kernel_size - 1) // 2

        height = y.size(2)
        width = y.size(3)

        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_q = torch.zeros_like(y)
        y_scales = torch.zeros_like(y)

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[0:1, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    context_prediction.weight,
                    bias=context_prediction.bias,
                )

                p = params[0:1, :, h:h + 1, w:w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1)) # 对于 compress() 是 self.entropy_parameters，是一个网络
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                y_crop = y_crop[0:1, :, padding:padding+1, padding:padding+1]
                y_crop_q = torch.round(y_crop - means_hat)
                y_hat[0, :, h + padding, w + padding] = (y_crop_q + means_hat)[0, :, 0, 0]
                y_q[0, :, h, w] = y_crop_q[0, :, 0, 0]
                y_scales[0, :, h, w] = scales_hat[0, :, 0, 0]
        # change to channel last
        y_q = y_q.permute(0, 2, 3, 1)
        y_scales = y_scales.permute(0, 2, 3, 1)
        y_string = self.gaussian_encoder.compress(y_q, y_scales)
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_string, y_hat

    def decompress_ar(self, y_string, channel, height, width, downsample, kernel_size,
                      context_prediction, params, entropy_parameters):
        device = next(self.parameters()).device
        padding = (kernel_size - 1) // 2

        y_size = get_downsampled_shape(height, width, downsample)
        y_height = y_size[0]
        y_width = y_size[1]

        y_hat = torch.zeros(
            (1, channel, y_height + 2 * padding, y_width + 2 * padding),
            device=params.device,
        )

        self.gaussian_encoder.set_stream(y_string)

        for h in range(y_height):
            for w in range(y_width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[0:1, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    context_prediction.weight,
                    bias=context_prediction.bias,
                )
                p = params[0:1, :, h:h + 1, w:w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)
                rv = self.gaussian_encoder.decode_stream(scales_hat)
                rv = rv.to(device)
                rv = rv + means_hat
                y_hat[0, :, h + padding: h + padding + 1, w + padding: w + padding + 1] = rv

        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_hat

    def compress(self, referframe, input_image): # 流程与 foward 相似
        device = input_image.device

        # [1] 光流估计
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv) # mvEncoder 实际是一个特征提取器？encode的含义
        z_mv = self.mvpriorEncoder(mvfeature)
        compressed_z_mv = torch.round(z_mv)

        # 实际压缩，forward 里并没有
        mv_z_string = self.bitEstimator_z_mv.compress(compressed_z_mv) # mv_z_string：压缩MV的latent vector
        mv_z_size = [compressed_z_mv.size(2), compressed_z_mv.size(3)]
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)

        # [2]
        params_mv = self.mvpriorDecoder(mv_z_hat)

        # [3] 实际压缩，forward 里并没有
        mv_y_string, mv_y_hat = self.compress_ar(mvfeature, 5, self.auto_regressive_mv, # mv_y_string：压缩MV
                                                 params_mv, self.entropy_parameters_mv)
        '''
        forward 中对应
        quant_mv = torch.round(mvfeature)
        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.entropy_parameters_mv(  
            torch.cat((params_mv, ctx_params_mv), dim=1)  
        )
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)
        '''

        # [4]
        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        temporal_prior_params = self.temporalPriorEncoder(context)
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1)) # 压缩
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)

        # [5]
        z_string = self.bitEstimator_z.compress(compressed_z) # 压缩后的 latent vector
        z_size = [compressed_z.size(2), compressed_z.size(3)]
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)

        # [6]
        params = self.priorDecoder(z_hat)

        # [7]
        y_string, y_hat = self.compress_ar(feature, 5, self.auto_regressive, # y_string：由熵编码再压缩
                                           torch.cat((temporal_prior_params, params), dim=1), self.entropy_parameters)

        # [8]
        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        # [10]
        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = len(y_string) * 8 / pixel_num
        bpp_z = len(z_string) * 8 / pixel_num
        bpp_mv_y = len(mv_y_string) * 8 / pixel_num
        bpp_mv_z = len(mv_z_string) * 8 / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "mv_y_string": mv_y_string,
                "mv_z_string": mv_z_string,
                "y_string": y_string,
                "z_string": z_string,
                }

    # 所以 mv_y_string, mv_z_string, y_string, z_string 就是压缩后的信息？
    def decompress(self, referframe, mv_y_string, mv_z_string, y_string, z_string, height, width):
        device = next(self.parameters()).device # 为什么需要这个
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        params_mv = self.mvpriorDecoder(mv_z_hat)
        mv_y_hat = self.decompress_ar(mv_y_string, self.out_channel_mv, height, width, 16, 5,
                                      self.auto_regressive_mv, params_mv,
                                      self.entropy_parameters_mv)

        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        temporal_prior_params = self.temporalPriorEncoder(context)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)
        params = self.priorDecoder(z_hat)
        y_hat = self.decompress_ar(y_string, self.out_channel_M, height, width, 16, 5,
                                   self.auto_regressive, torch.cat((temporal_prior_params, params), dim=1),
                                   self.entropy_parameters)
        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))
        recon_image = recon_image.clamp(0, 1)

        return recon_image

    def forward(self, referframe, input_image):
        # print("forward")
        # print(referframe.shape)
        # print(input_image.shape)
        # Figure 2 的橙色部分
        # [1]
        estmv = self.opticFlow(input_image, referframe) # 光流估计运动向量
        mvfeature = self.mvEncoder(estmv)   # 为什么编码后作为特征？mvEncoder做了什么？
        z_mv = self.mvpriorEncoder(mvfeature)   # z 代表什么？ —— latent vector？z 常表示潜在特征空间
        compressed_z_mv = torch.round(z_mv) # 量化：或许就如命名 compressed 一样，此处量化只是减小空间开销？
        
        # [2]
        params_mv = self.mvpriorDecoder(compressed_z_mv) # 解码，为什么要 编码+量化+解码？为了引入量化？为了压缩？
        # params_mv，作为什么的参数？如果 编码+量化+解码 只是为了引入量化，那么params_mv地位应该等同于mvfeature，但之后又单独量化了mvfeature
        
        # [3]
        quant_mv = torch.round(mvfeature)   # 单独量化 mvfeature

        ctx_params_mv = self.auto_regressive_mv(quant_mv) # 自回归编码，为什么要自回归编码？
        # 为mv的每个值引入上下文信息？
        gaussian_params_mv = self.entropy_parameters_mv(  # 
            torch.cat((params_mv, ctx_params_mv), dim=1)  # 拼接 mvPriorDecoder([mvPriorEncoder(mvfeature)]) 和 auto_regressive([mvfeature]) 
        )   # 熵编码的定义：对于一个给定的信息源，用最少的比特数来编码这个信息源；输入：信息源；输出：编码后的信息
        # 熵编码与分布的关系：熵编码将出现概率大的信息用较短的编码表示，出现概率小的信息用较长的编码表示，而概率分布描述了信息出现的概率
        # 此处不是进行熵编码，而是进行熵参数的估计，即对于给定的信息源，估计其概率分布
        # 这里假设 mvfeature 的分布是高斯分布？用特征 预测 高斯分布的参数
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1) # 从预测的参数中分离出均值和标准差（位置和规模）

        # [4]
        quant_mv_upsample = self.mvDecoder_part1(quant_mv) # 上采样，为了恢复到合适的分辨率（原始尺寸）进行运动补偿？
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample) # 注意是mv_refine而不是context_refine
        # 是因为单单对mv进行上采样信息不够？mv_refine的卷积学到的是什么？
        context = self.motioncompensation(referframe, quant_mv_upsample_refine) # 此处motioncompensation包含了原文的feature extractor和context refine
        # 虽然叫motioncompensation，但除了运动补偿还包含了原文的feature extractor和context refine，
        # 除开二者，所以flow_warp对应原文的warp？

        # 熵模型部分 （Figure 3）
        # 得到 temporal_prior_params
        temporal_prior_params = self.temporalPriorEncoder(context) # 因为 context 是运动补偿后的结果（有光流带来的帧间关系），所以这里是时域先验？
        # 得到 params 除去了 temporal_prior_params 和 ctx_params，所以此处对应原文的 hyperprior？
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z) # 量化后都称为compressed？

        # [6]
        params = self.priorDecoder(compressed_z) # 得到 ctx_params

        # [7]
        feature_renorm = feature # ? renorm，就像mvfeature一样，单独又量化了一次

        compressed_y_renorm = torch.round(feature_renorm)

        ctx_params = self.auto_regressive(compressed_y_renorm) # 自回归引入空间上下文信息
        # 综合使用 temporal_prior_params, params, ctx_params 估计分布的参数
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        # [8]
        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))
        # 如果用 compressed_y_renorm 和 context 就可以重建图像，那为什么要估计 means_hat 和 scales_hat？

        # [9]
        total_bits_y, _ = self.feature_probs_based_sigma(  # 用高斯分布的参数估计压缩后的比特数（没有真正压缩)
            feature_renorm, means_hat, scales_hat)
        total_bits_mv, _ = self.feature_probs_based_sigma(mvfeature, means_hat_mv, scales_hat_mv)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)
        total_bits_z_mv, _ = self.iclr18_estrate_bits_z_mv(compressed_z_mv)

        # [10]
        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_mv_z = total_bits_z_mv / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "context": context,
                "x_tilde": flow_warp(referframe, quant_mv_upsample_refine),
                }
    
    def step1_forward(self, referframe, input_image):
        estmv = self.opticFlow(input_image, referframe) 
        mvfeature = self.mvEncoder(estmv)  
        quant_mv = torch.round(mvfeature) 
        quant_mv_upsample = self.mvDecoder_part1(quant_mv)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample) # 此即 mv_decoder_part2
        x_tilde = flow_warp(referframe, quant_mv_upsample_refine)

        z_mv = self.mvpriorEncoder(mvfeature) 
        compressed_z_mv = torch.round(z_mv) 
        params_mv = self.mvpriorDecoder(compressed_z_mv)

        quant_mv = torch.round(mvfeature) 
        ctx_params_mv = self.auto_regressive_mv(quant_mv) 
        gaussian_params_mv = self.entropy_parameters_mv( 
            torch.cat((params_mv, ctx_params_mv), dim=1)  
        ) 
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)

        total_bits_mv, _ = self.feature_probs_based_sigma(mvfeature, means_hat_mv, scales_hat_mv)
        total_bits_z_mv, _ = self.iclr18_estrate_bits_z_mv(compressed_z_mv)
        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_mv_z = total_bits_z_mv / pixel_num

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "x_tilde": x_tilde,
                }
    
    def forward_exclude_me(self, referframe, input_image):
        # 参考原文，为了排除运动补偿部分，将原 reference frame 作为 context
        context = referframe
        temporal_prior_params = self.temporalPriorEncoder(context)

        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        feature_renorm = feature
        recon_image_feature = self.contextualDecoder_part1(feature_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)
        params = self.priorDecoder(compressed_z)

        compressed_y_renorm = torch.round(feature_renorm)
        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )

        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        total_bits_y, _ = self.feature_probs_based_sigma(feature_renorm, means_hat, scales_hat)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp = bpp_y + bpp_z

        return {"bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                }
    

    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict)
