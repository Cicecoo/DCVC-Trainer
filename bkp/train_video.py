# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import random
import shutil
import sys

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

# from compressai.datasets import VideoFolder
from compressai.optimizers import net_aux_optimizer
# from compressai.zoo import video_models

from dvc_dataset import DataSet
from src.models.DCVC_net import DCVC_net


def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0

    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target, epoch):
        # assert isinstance(target, type(output["x_hat"]))
        # assert len(output["x_hat"]) == len(target)

        # self._check_tensors_list(target)
        # self._check_tensors_list(output["x_hat"])
        # output[x_hat] = [output[x_hat]]
        # target = [target]

        # _, _, H, W = target[0].size()
        # num_frames = len(target)
        out = {}
        # num_pixels = H * W * num_frames

        # # Get scaled and raw loss distortions for each frame
        # scaled_distortions = []
        # distortions = []
        # for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
        #     scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

        #     distortions.append(distortion)
        #     scaled_distortions.append(scaled_distortion)

        #     if self.return_details:
        #         out[f"frame{i}.mse_loss"] = distortion
        # # aggregate (over batch and frame dimensions).
        # out["mse_loss"] = torch.stack(distortions).mean()

        scaled_distortion, distortion = self._get_scaled_distortion(output["x_hat"], target)
        out["mse_loss"] = distortion.mean()

        # average scaled_distortions accros the frames
        # scaled_distortions = sum(scaled_distortions) / num_frames

        # assert isinstance(output["likelihoods"], list)
        # likelihoods_list = output.pop("likelihoods")

        # # collect bpp info on noisy tensors (estimated differentiable entropy)
        # bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        # if self.return_details:
        #     out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...

        # # now we either use a fixed lambda or try to balance between 2 lambdas
        # # based on a target bpp.
        # lambdas = torch.full_like(bpp_loss, self.lmbda)

        # bpp_loss = bpp_loss.mean()

        if epoch < 1:
            bpp_loss = output["bpp_mv_y"] + output["bpp_mv_z"]
        elif epoch < 4:
            bpp_loss = 0
        elif epoch < 7:
            bpp_loss = output["bpp_y"] + output["bpp_z"]
        else:
            bpp_loss = output["bpp_y"] + output["bpp_z"] + output["bpp_mv_y"] + output["bpp_mv_z"]

        # print("scaled_distortion: ", scaled_distortion, "distortion: ", distortion)

        out["loss"] = self.lmbda *  out["mse_loss"]  + bpp_loss
        # print(f"loss: {out['loss']}, scaled_distortion: {scaled_distortion}, bpp_loss: {bpp_loss}")
        out["distortion"] = sum(scaled_distortion)/4
        out["bpp_loss"] = bpp_loss
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss

        if backward is True:
            aux_loss.backward()

    return aux_loss_sum


# def configure_optimizers(net, args):
#     """Separate parameters for the main optimizer and the auxiliary optimizer.
#     Return two optimizers"""
#     conf = {
#         "net": {"type": "Adam", "lr": args.learning_rate},
#         "aux": {"type": "Adam", "lr": args.aux_learning_rate},
#     }
#     optimizer = net_aux_optimizer(net, conf)
#     return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer,  epoch, clip_max_norm
):# aux_optimizer,
    model.train()
    device = next(model.parameters()).device

    for i, batch in enumerate(train_dataloader):
        # d = [frames.to(device) for frames in batch]
        input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch
        input_image = input_image.to(device)
        ref_image = ref_image.to(device)

        optimizer.zero_grad()
        # aux_optimizer.zero_grad()

        out_net = model(ref_image, input_image)

        out_criterion = criterion(out_net, input_image, epoch)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        # aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                # f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"]} |'
                # f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    # aux_loss = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch
            input_image = input_image.to(device)
            ref_image = ref_image.to(device)
            
            out_net = model(ref_image, input_image)
            out_criterion = criterion(out_net, input_image, epoch)

            # aux_loss.update(compute_aux_loss(model.aux_loss()))
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        # f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")

class Args:
    def __init__(self):
        self.model = 'ssf2020'
        self.dataset = 'H:/Data/vimeo_septuplet/vimeo_septuplet/sequences/'

        self.epochs = 10
        self.batch_size = 4
        self.num_workers = 4
        self.test_batch_size = 1

        self.lmbda = 256 #1e-2
        self.learning_rate = 1e-4
        self.aux_learning_rate = 1e-3
        self.patch_size = (256, 256)

        self.cuda = True
        self.save = True
        self.seed = None
        self.clip_max_norm = 1.0
        self.checkpoint = None

def main(argv):
    args = Args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # # Warning, the order of the transform composition should be kept.
    # train_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    # )

    # test_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
    # )

    # train_dataset = VideoFolder(
    #     args.dataset,
    #     rnd_interval=True,
    #     rnd_temp_order=True,
    #     split="train",
    #     transform=train_transforms,
    # )
    # test_dataset = VideoFolder(
    #     args.dataset,
    #     rnd_interval=False,
    #     rnd_temp_order=False,
    #     split="test",
    #     transform=test_transforms,
    # )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataset = DataSet(path='/mnt/h/Data/vimeo_septuplet/vimeo_septuplet/mini_dvc_test.txt', im_height=256, im_width=256)
    test_dataset = DataSet(path='/mnt/h/Data/vimeo_septuplet/vimeo_septuplet/mini_dvc_test_val.txt', im_height=256, im_width=256)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # net = video_models[args.model](quality=3)
    net = DCVC_net()
    net = net.to(device)

    # optimizer, aux_optimizer = configure_optimizers(net, args)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda, return_details=True)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            # aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            # save_checkpoint(
            #     {
            #         "epoch": epoch,
            #         "state_dict": net.state_dict(),
            #         "loss": loss,
            #         "optimizer": optimizer.state_dict(),
            #         # "aux_optimizer": aux_optimizer.state_dict(),
            #         "lr_scheduler": lr_scheduler.state_dict(),
            #     },
            #     is_best,
            # )
            torch.save(net.state_dict(), "checkpoint.pth")

if __name__ == "__main__":
    main(sys.argv[1:])