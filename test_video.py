# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
import os
import concurrent.futures
import multiprocessing
import torch
import json
import numpy as np
from PIL import Image
from src.models.DCVC_net import DCVC_net
from src.zoo.image import model_architectures as architectures # 包括了 Cheng2020Anchor 和 FactorizedPrior, ScaleHyperprior, MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors
import time
from tqdm import tqdm
import warnings
from pytorch_msssim import ms_ssim

# import onnx


warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--i_frame_model_name', type=str, default="cheng2020-anchor") 
    parser.add_argument('--i_frame_model_path', type=str, nargs="+")
    parser.add_argument('--model_path',  type=str, nargs="+")
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--write_recon_frame", type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--recon_bin_path', type=str, default="recon_bin_path")
    parser.add_argument('--output_json_result_path', type=str, required=True)
    parser.add_argument("--model_type",  type=str,  default="psnr", help="psnr, msssim")

    args = parser.parse_args()
    return args


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


def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0)*255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)


def encode_one(args_dict, device):
    i_frame_load_checkpoint = torch.load(args_dict['i_frame_model_path'], # 从 default="cheng2020-anchor" 来看应该是指 intra codeing
                                         map_location=torch.device('cpu'))
    i_frame_net = architectures[args_dict['i_frame_model_name']].from_state_dict(
        i_frame_load_checkpoint).eval()

    video_net = DCVC_net() # 与 i_frame_net 呼应的帧间编码？
    load_checkpoint = torch.load(args_dict['model_path'], map_location=torch.device('cpu'))
    video_net.load_dict(load_checkpoint)

    video_net = video_net.to(device)
    video_net.eval()

    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    if args_dict['write_stream']: # write_stream 为什么和 update 有关？
        video_net.update(force=True) # 比起 更新，实际更接近初始化？
        i_frame_net.update(force=True)
        
    '''
    video_net 的 update 函数：
        
    def update(self, force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    i_frame_net （此处为 Cheng2020Anchor）的 update 函数：
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    似乎都是熵模型相关
    '''

    # # 在 onnx 中注册算子
    # torch.onnx.register_custom_op_symbolic('dummy::grid_sampler', bilinear_grid_sample, 11)
    # torch.onnx.export(
    #     video_net, 
    #     (torch.randn(1, 3, 1920, 1024).to(device), 
    #      torch.randn(1, 3, 1920, 1024).to(device)), 
    #     "video_net.onnx", 
    #     opset_version=16, 
    #     )
    # # torch.onnx.export(i_frame_net, torch.randn(1, 3, 1920, 1024).to(device), "i_frame_net.onnx")
    # exit()

    sub_dir_name = args_dict['video_path']
    ref_frame = None
    frame_types = [] # 看过 h264 后理解，用于记录帧类型（I、P），给出解码顺序
    qualitys = [] # PSNR 或 MS-SSIM，给出每帧的压缩？质量
    bits = [] # 对于 I 帧，记录 bit.item()；对于 P 帧，bpp.item()*frame_pixel_num。即每帧的比特数
    bits_mv_y = [] # mv应该是运动向量，y 是？
    bits_mv_z = []
    bits_y = [] # y、z 是？
    bits_z = []

    gop_size = args_dict['gop'] # 固定的 GOP 大小？
    frame_pixel_num = 0 
    frame_num = args_dict['frame_num'] # 在 main 计算得到

    recon_bin_folder = os.path.join(args_dict['recon_bin_path'], sub_dir_name,
                                    os.path.basename(args_dict['model_path'])[:-4])
    if not os.path.exists(recon_bin_folder): # recon 指 reconstruction？
        os.makedirs(recon_bin_folder)

    # Figure out the naming convention # 为什么要确定命名规则？此处的两种规则是某种约定？
    pngs = os.listdir(os.path.join(args_dict['dataset_path'], sub_dir_name))
    if 'im1.png' in pngs:
        padding = 1
    elif 'im00001.png' in pngs: # 用ffmpeg得到的命名与此相同
        padding = 5
    else:
        raise ValueError('unknown image naming convention; please specify')

    with torch.no_grad():
        for frame_idx in range(frame_num):
            # 读取一帧
            ori_frame = read_frame_to_torch(
                os.path.join(args_dict['dataset_path'],
                             sub_dir_name,
                             f"im{str(frame_idx+1).zfill(padding)}.png"))
            ori_frame = ori_frame.to(device) # 保证设备一致

            if frame_pixel_num == 0: # frame_pixel_num 只初始化一次
                frame_pixel_num = ori_frame.shape[2]*ori_frame.shape[3] # shape 为 [batch, channel, height, width]， 见 read_frame_to_torch 
            else: # 每帧 比较前后帧尺寸（保证一致）
                assert(frame_pixel_num == ori_frame.shape[2]*ori_frame.shape[3])

            if args_dict['write_stream']:
                bin_path = os.path.join(recon_bin_folder, f"{frame_idx}.bin") # 写位置
                if frame_idx % gop_size == 0:
                    # 推理 帧内压缩
                    result = i_frame_net.encode_decode(ori_frame, bin_path)
                    # 更新参考帧 用x_hat作为新的参考帧（I帧）
                    ref_frame = result["x_hat"]
                    # 记录 bpp
                    bpp = result["bpp"]
                    # 标记I帧
                    frame_types.append(0)
                    # 计算并记录性能指标
                    bits.append(bpp*frame_pixel_num)
                    bits_mv_y.append(0)
                    bits_mv_z.append(0)
                    bits_y.append(0)
                    bits_z.append(0)
                else:
                    # 推理 DCVC
                    result = video_net.encode_decode(ref_frame, ori_frame, bin_path)
                    # 更新同上
                    ref_frame = result['recon_image']
                    bpp = result['bpp']
                    frame_types.append(1)
                    bits.append(bpp*frame_pixel_num)
                    bits_mv_y.append(result['bpp_mv_y']*frame_pixel_num)
                    bits_mv_z.append(result['bpp_mv_z']*frame_pixel_num)
                    bits_y.append(result['bpp_y']*frame_pixel_num)
                    bits_z.append(result['bpp_z']*frame_pixel_num)
            else:
                # 只有 I、P 两种帧？
                if frame_idx % gop_size == 0: # I 帧
                    result = i_frame_net(ori_frame)
                    bit = sum((torch.log(likelihoods).sum() / (-math.log(2)))  # bit = Sigma(Sigma(log(likelihoods))) / (-log(2)) = - Sigma Sigma(log2(likelihoods))
                              for likelihoods in result["likelihoods"].values()) # likelihoods 是什么样子（为什么是二维）？
                    ref_frame = result["x_hat"] # 为 P 帧初始化参考帧
                    
                    frame_types.append(0)
                    bits.append(bit.item())
                    bits_mv_y.append(0)
                    bits_mv_z.append(0)
                    bits_y.append(0)
                    bits_z.append(0)
                else: # P 帧
                    result = video_net(ref_frame, ori_frame)
                    ref_frame = result['recon_image'] # 由于没有B帧，下一帧如果需要参考帧，一定是上一帧，对于 P 帧是重建后的帧？
                    
                    bpp = result['bpp']
                    frame_types.append(1)
                    bits.append(bpp.item()*frame_pixel_num)
                    bits_mv_y.append(result['bpp_mv_y'].item()*frame_pixel_num)
                    bits_mv_z.append(result['bpp_mv_z'].item()*frame_pixel_num)
                    bits_y.append(result['bpp_y'].item()*frame_pixel_num)
                    bits_z.append(result['bpp_z'].item()*frame_pixel_num)

            ref_frame = ref_frame.clamp_(0, 1) # clamp 限幅
            if args_dict['write_recon_frame']:
                write_torch_frame(ref_frame.squeeze(), os.path.join(recon_bin_folder, f"recon_frame_{frame_idx}.png"))
            if args_dict['model_type'] == 'psnr':
                qualitys.append(PSNR(ref_frame, ori_frame))
            else:
                qualitys.append(
                    ms_ssim(ref_frame, ori_frame, data_range=1.0).item())

    cur_all_i_frame_bit = 0 # 当前所有 I 帧的比特数之和
    cur_all_i_frame_quality = 0
    cur_all_p_frame_bit = 0
    cur_all_p_frame_bit_mv_y = 0
    cur_all_p_frame_bit_mv_z = 0
    cur_all_p_frame_bit_y = 0
    cur_all_p_frame_bit_z = 0
    cur_all_p_frame_quality = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_all_i_frame_bit += bits[idx]
            cur_all_i_frame_quality += qualitys[idx]
            cur_i_frame_num += 1
        else:
            cur_all_p_frame_bit += bits[idx]
            cur_all_p_frame_bit_mv_y += bits_mv_y[idx]
            cur_all_p_frame_bit_mv_z += bits_mv_z[idx]
            cur_all_p_frame_bit_y += bits_y[idx]
            cur_all_p_frame_bit_z += bits_z[idx]
            cur_all_p_frame_quality += qualitys[idx]
            cur_p_frame_num += 1

    log_result = {}
    log_result['name'] = f"{os.path.basename(args_dict['model_path'])}_{sub_dir_name}"
    log_result['ds_name'] = args_dict['ds_name']
    log_result['video_path'] = args_dict['video_path']
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = cur_i_frame_num
    log_result['p_frame_num'] = cur_p_frame_num
    log_result['ave_i_frame_bpp'] = cur_all_i_frame_bit / cur_i_frame_num / frame_pixel_num
    log_result['ave_i_frame_quality'] = cur_all_i_frame_quality / cur_i_frame_num
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = cur_all_p_frame_bit / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_y'] = cur_all_p_frame_bit_mv_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_z'] = cur_all_p_frame_bit_mv_z / total_p_pixel_num
        log_result['ave_p_frame_bpp_y'] = cur_all_p_frame_bit_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_z'] = cur_all_p_frame_bit_z / total_p_pixel_num
        log_result['ave_p_frame_quality'] = cur_all_p_frame_quality / cur_p_frame_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_quality'] = 0
        log_result['ave_p_frame_bpp_mv_y'] = 0
        log_result['ave_p_frame_bpp_mv_z'] = 0
        log_result['ave_p_frame_bpp_y'] = 0
        log_result['ave_p_frame_bpp_z'] = 0
    log_result['ave_all_frame_bpp'] = (cur_all_i_frame_bit + cur_all_p_frame_bit) / \
        (frame_num * frame_pixel_num)
    log_result['ave_all_frame_quality'] = (cur_all_i_frame_quality + cur_all_p_frame_quality) / frame_num
    return log_result


def worker(use_cuda, args):
    # 确定性相关 保证可复现
    if args['write_stream']:
        torch.backends.cudnn.benchmark = False
        if 'use_deterministic_algorithms' in dir(torch): # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
            torch.use_deterministic_algorithms(True)
        else:
            torch.set_deterministic(True)
    torch.manual_seed(0) # 固定随机种子
    torch.set_num_threads(1) 
    np.random.seed(seed=0)

    # 设备（并行）相关
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    # encode_one 的 one 指的应该是一段视频（序列），而非一帧
    result = encode_one(args, device)
    result['model_idx'] = args['model_idx']
    return result


def filter_dict(result):
    keys = ['i_frame_num', 'p_frame_num', 'ave_i_frame_bpp', 'ave_i_frame_quality', 'ave_p_frame_bpp',
            'ave_p_frame_bpp_mv_y', 'ave_p_frame_bpp_mv_z', 'ave_p_frame_bpp_y',
            'ave_p_frame_bpp_z', 'ave_p_frame_quality', 'ave_all_frame_bpp', 'ave_all_frame_quality']
    res = {k: v for k, v in result.items() if k in keys}
    return res


def main():
    torch.backends.cudnn.enabled = True
    args = parse_args()
    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f: # test_config 此处是 dataset_config.json
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    count_frames = 0
    count_sequences = 0
    begin_time = time.time()
    for ds_name in config: # ds_name 为数据集名称 dataset name
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for model_idx in range(len(args.model_path)): # 用来连续测试多种模型？
                cur_dict = {}
                cur_dict['model_idx'] = model_idx
                cur_dict['i_frame_model_path'] = args.i_frame_model_path[model_idx]
                cur_dict['i_frame_model_name'] = args.i_frame_model_name
                cur_dict['model_path'] = args.model_path[model_idx]
                cur_dict['video_path'] = seq_name
                cur_dict['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                cur_dict['frame_num'] = config[ds_name]['sequences'][seq_name]['frames'] 
                cur_dict['dataset_path'] = config[ds_name]['base_path']
                cur_dict['write_stream'] = args.write_stream
                cur_dict['write_recon_frame'] = args.write_recon_frame
                cur_dict['recon_bin_path'] = args.recon_bin_path
                cur_dict['model_type'] = args.model_type
                cur_dict['ds_name'] = ds_name

                count_frames += cur_dict['frame_num']

                obj = threadpool_executor.submit(
                    worker,
                    args.cuda,
                    cur_dict)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}

    for ds_name in config:
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for model_idx in range(len(args.model_path)):
                ckpt = os.path.basename(args.model_path[model_idx])
                for res in results:
                    if res['name'].startswith(ckpt) and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][ckpt] = filter_dict(res)

    with open(args.output_json_result_path, 'w') as fp:
        json.dump(log_result, fp, indent=2)

    total_minutes = (time.time() - begin_time) / 60

    count_models = len(args.model_path)
    count_frames = count_frames // count_models
    print('Test finished')
    print(f'Tested {count_models} models on {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
