import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
# from ..utils.stream_helper import get_downsampled_shape
from ..layers.layers import MaskedConv2d, subpel_conv3x3
from .DVC_Spynet import ME_Spynet as DVC_Spynet # modified

# def save_model(model, iter):
#     torch.save(model.state_dict(), "./snapshot_241231/iter{}.model".format(iter))

# def load_model(model, f):
#     print("load DCVC format")
#     with open(f, 'rb') as f:
#         pretrained_dict = torch.load(f)
#         model_dict = model.state_dict()
#         # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         result_dict = {}
#         i = 0
#         v = list(pretrained_dict.values())
#         for key, weight in model_dict.items():
#             result_key = key
#             result_dict[result_key] = v[i]
#             i += 1
#         model_dict.update(result_dict)
#         model.load_state_dict(model_dict)

#     f = str(f)
#     if f.find('iter') != -1 and f.find('.model') != -1:
#         st = f.find('iter') + 4
#         ed = f.find('.model', st)
#         return int(f[st:ed])    #return step
#     else:
#         return 0

class DCVC_net(nn.Module):
    def __init__(self):
        super().__init__()
        out_channel_mv = 128
        out_channel_N = 64
        out_channel_M = 96
        self.out_channel_mv = out_channel_mv
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.bitEstimator_z = BitEstimator(out_channel_N)
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
        self.mvEncoder = MvEncoder_net(in_channel=2, out_channel=out_channel_mv)
        self.mvDecoder_part1 = MvDecoder_part1_net(in_channel=out_channel_mv, out_channel=2)
        self.mvDecoder_part2 = MvDecoder_part2_net()

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

        self.priorEncoder = PriorEncoder_net(out_channel_M, out_channel_N)
        self.priorDecoder = PriorDecoder_net(in_channel=out_channel_N, out_channel=out_channel_M)

        self.mvpriorEncoder = MvpriorEncoder_net(out_channel_mv, out_channel_N)
        self.mvpriorDecoder = MvpriorDecoder_net(out_channel_N, out_channel_mv)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
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
        self.temporalPriorEncoder = TemporalPriorEncoder_net(out_channel_N, out_channel_M)

        # self.opticFlow = ME_Spynet()
        self.opticFlow = DVC_Spynet()

        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

    def motioncompensation(self, ref, mv):
        ref_feature =  self.feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context =  self.context_refine(prediction_init)

        return context

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
        # outputs = self.quantize(
        #     feature, "dequantize", mean
        # )
        outputs = feature
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
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

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None, quant_noise_z_mv=None):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)

        if self.training:
            compressed_z_mv = z_mv + quant_noise_z_mv
        else:
            compressed_z_mv = torch.round(z_mv)
        
        params_mv = self.mvpriorDecoder(compressed_z_mv)

        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)

        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)

        quant_mv_upsample = self.mvDecoder_part1(quant_mv)

        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)

        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        # predict_pixel = flow_warp(referframe, quant_mv_upsample_refine)

        temporal_prior_params = self.temporalPriorEncoder(context)

        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        
        params = self.priorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_y_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_y_renorm = torch.round(feature_renorm)


        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )

        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context) , dim=1))
        clipped_recon_image = recon_image.clamp(0., 1.)

        total_bits_y, _ = self.feature_probs_based_sigma(compressed_y_renorm, means_hat, scales_hat)
        total_bits_mv, _ = self.feature_probs_based_sigma(quant_mv, means_hat_mv, scales_hat_mv)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)
        total_bits_z_mv, _ = self.iclr18_estrate_bits_z_mv(compressed_z_mv)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_mv_z = total_bits_z_mv / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

# distortion
        # distortion1 = torch.mean((predict_pixel - input_image).pow(2))
        # distortion2 = torch.mean((recon_image - input_image).pow(2))

        # return clipped_recon_image, distortion1, distortion2, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp
        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "clipped_recon_image": clipped_recon_image,
                "context": context,
                "x_tilde": flow_warp(referframe, quant_mv_upsample_refine),
                }
    
    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict)


# =============================================================================================
# =============================================================================================
# =============================================================================================

class MvEncoder_net(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self, in_channel, out_channel):
        super(MvEncoder_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel)
        self.conv4 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel + out_channel) / (out_channel + out_channel))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)

class MvDecoder_part1_net(nn.Module):
    '''
    Decode residual
    '''
    def __init__(self,in_channel,out_channel):
        super(MvDecoder_part1_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(in_channel, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(in_channel, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(in_channel, inverse=True)
        #self.deconv4 = nn.ConvTranspose2d(in_channel, 3, 5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        
    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x

class MvDecoder_part2_net(nn.Module):
    def __init__(self):
        super(MvDecoder_part2_net, self).__init__()
        self.l1 = nn.Conv2d(5, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l1.weight.data)
        torch.nn.init.constant_(self.l1.bias.data, 0.0)
        self.r1 = nn.LeakyReLU(negative_slope=0.1)
        self.l2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l2.weight.data)
        torch.nn.init.constant_(self.l2.bias.data, 0.0)
        self.r2 = nn.LeakyReLU(negative_slope=0.1)
        self.l3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l3.weight.data)
        torch.nn.init.constant_(self.l3.bias.data, 0.0)
        self.r3 = nn.LeakyReLU(negative_slope=0.1)
        self.l4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l4.weight.data)
        torch.nn.init.constant_(self.l4.bias.data, 0.0)
        self.r4 = nn.LeakyReLU(negative_slope=0.1)
        self.l5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l5.weight.data)
        torch.nn.init.constant_(self.l5.bias.data, 0.0)
        self.r5 = nn.LeakyReLU(negative_slope=0.1)
        self.l6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l6.weight.data)
        torch.nn.init.constant_(self.l6.bias.data, 0.0)
        self.r6 = nn.LeakyReLU(negative_slope=0.1)
        self.l7 = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.l7.weight.data)
        torch.nn.init.constant_(self.l7.bias.data, 0.0)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.r3(self.l3(x))
        x = self.r4(self.l4(x))
        x = self.r5(self.l5(x))
        x = self.r6(self.l6(x))
        x = self.l7(x)
        return x

class PriorEncoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorEncoder_net, self).__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class PriorDecoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorDecoder_net, self).__init__()
        self.l1 = nn.ConvTranspose2d(in_channel, out_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.ConvTranspose2d(out_channel, out_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (out_channel + out_channel))))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class MvpriorEncoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MvpriorEncoder_net, self).__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class MvpriorDecoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MvpriorDecoder_net, self).__init__()
        self.l1 = nn.ConvTranspose2d(in_channel, in_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.ConvTranspose2d(in_channel, in_channel * 3 // 2, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2*(in_channel + in_channel * 3 // 2)/(2*in_channel))))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.ConvTranspose2d(in_channel * 3 // 2, out_channel*2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2*(in_channel * 3 // 2 + out_channel*2)/(2 * in_channel * 3 // 2))))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class Entropy_parameters_net(nn.Module):
    def __init__(self, channel):
        super(Entropy_parameters_net, self).__init__()
        self.l1 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class TemporalPriorEncoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalPriorEncoder_net, self).__init__()
        self.l1 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = GDN(in_channel)
        self.l2 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = GDN(in_channel)
        self.l3 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
        self.r3 = GDN(in_channel)
        self.l4 = nn.Conv2d(in_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l4.weight.data, (math.sqrt(2*(in_channel+out_channel)/(2*in_channel))))
        torch.nn.init.constant_(self.l4.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.r3(self.l3(x))
        x = self.l4(x)
        return x
