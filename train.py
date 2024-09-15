
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from src.models.image_model import IntraNoAR
from src.models.video_model import DMC
from src.utils.common import AverageMeter
import torchvision

import random


def get_stage_config(current_epoch):
    # borders_of_stages = [1, 4, 7, 10, 16, 21, 24, 25, 27, 30] # Default
    borders_of_stages = [1, 4, 7, 10, 21, 26, 29, 30, 32, 35] # early More
    # borders_of_stages = [1, 4, 7, 10, 16, 21, 27, 29, 35, 38] # More
    # borders_of_stages = [1, 4, 7, 10, 16, 21, 28, 30, 30, 30] # no avg_loss
    # borders_of_stages = [0, 0, 0, 0, 0, 0, 5, 7, 10, 11]    # Fine-tuning final stages
    # borders_of_stages = [0, 0, 0, 0, 0, 0, 8, 10, 11, 11]    # Fine-tuning no avg_loss (with total_epochs=10)

    if current_epoch < borders_of_stages[0]:
        stage = 0
    elif current_epoch < borders_of_stages[1]:
        stage = 1
    elif current_epoch < borders_of_stages[2]:
        stage = 2
    elif current_epoch < borders_of_stages[3]:
        stage = 3
    elif current_epoch < borders_of_stages[4]:
        stage = 4
    elif current_epoch < borders_of_stages[5]:
        stage = 5
    elif current_epoch < borders_of_stages[6]:
        stage = 6
    elif current_epoch < borders_of_stages[7]:
        stage = 7
    elif current_epoch < borders_of_stages[8]:
        stage = 8
    elif current_epoch < borders_of_stages[9]:
        stage = 9
        
    # stage: 0 - 9
    config = {}
    config["loss"] = []
    config["stage"] = stage

    # 1. Number of Frames
    if stage < 5:
        config["nframes"] = 2
    elif stage < 6:
        config["nframes"] = 3
    elif stage < 10:
        config["nframes"] = 5

    # 2. Loss
    if stage < 1:
        config["loss"].append("mv_dist")
    elif stage < 2:
        config["loss"].append("mv_dist")
        config["loss"].append("mv_rate")
    elif stage < 3:
        config["loss"].append("x_dist")
    elif stage < 4:
        config["loss"].append("x_dist")
        config["loss"].append("x_rate")
    elif stage < 10:
        config["loss"].append("x_dist")
        config["loss"].append("x_rate")
        config["loss"].append("mv_rate")

    # 3. Learning Rate
    if stage < 7:
        config["lr"] = 1e-4
    elif stage < 8:
        config["lr"] = 1e-5
    elif stage < 9:
        config["lr"] = 5e-5
    elif stage < 10:
        config["lr"] = 1e-5

    # 4. Loss Avg.
    if stage < 8:
        config["avg_loss"] = False
    elif stage < 10:
        config["avg_loss"] = True

    # 5. mode.
    if stage < 2:
        config["mode"] = "inter"
    elif stage < 4:
        config["mode"] = "recon"
    elif stage < 10:
        config["mode"] = "all"
    return config


class DCLightning(LightningModule):
    def __init__(
        self, kwargs
    ):
        # ----------------- Single P-frame --------------------
        # Stage 0: mv_dist                                      < 1
        # Stage 1: mv_dist & mv_rate                            < 4
        # Stage 2: x_dist                                       < 7
        # Stage 3: x dist & x_rate                              < 10
        # Stage 4: x dist & x_rate & mv_rate                    < 16

        # ----------------- Dual P-frame --------------------
        # Stage 5: x dist & x_rate & mv_rate                    < 21

        # ----------------- Four P-frame --------------------
        # Stage 6: x dist & x_rate & mv_rate                    < 24
        # Stage 7: x dist & x_rate & mv_rate (1e-5)             < 25
        # Stage 8: x dist & x_rate & mv_rate (5e-5) (avg_loss)  < 27
        # Stage 9: x dist & x_rate & mv_rate (1e-5) (avg_loss)  < 30

        super().__init__()
        self.i_frame_model = self.load_i_frame_model()
        self.p_frame_model = DMC()

        self.q_index_to_lambda = {
            # 0: 340,
            # 1: 680,
            # 2: 1520,
            # 3: 3360,
            0: 85,
            1: 170,
            2: 380,
            3: 840,
        }
        self.weights = [0.5, 1.2, 0.5, 0.9]
        self.automatic_optimization = False
        self.single = kwargs["single"]
        self.quality = kwargs["quality"]
        
    # (out_net, frames[i+1], q_index=q, objective=objective)
    def rate_distortion_loss(
        self,
        out_net,
        target,
        q_index: int,
        objective: list,
        frame_idx: int,
    ):
        bpp = torch.tensor(0.0).to(out_net["dpb"]["ref_frame"].device)
        if "mv_rate" in objective:
            bpp += out_net["bpp_mv_y"] + out_net["bpp_mv_z"]
        if "x_rate" in objective:
            bpp += out_net["bpp_y"] + out_net["bpp_z"]

        out = {"bpp": bpp}
        out["mse"] = F.mse_loss(out_net["dpb"]["ref_frame"], target)
        out["psnr"] = 10 * torch.log10(1 * 1 / out["mse"])

        if self.use_weighted_loss:
            out["loss"] = (
                self.q_index_to_lambda[q_index] * out["mse"] * self.weights[frame_idx]
                + out["bpp"]
            )
        else:
            out["loss"] = self.q_index_to_lambda[q_index] * out["mse"] + out["bpp"]
        return out

    def update(self, force=True):
        return self.model.update(force=force)

    def compress(self, ref_frame, x):
        return self.model.compress(ref_frame, x, self.quality)

    def decompress(
        self, ref_frame, mv_y_string, mv_z_string, y_string, z_string, height, width
    ):
        return self.model.decompress(
            ref_frame, y_string, z_string, mv_y_string, mv_z_string, height, width
        )

    def training_step(self, batch, batch_idx):
        config = get_stage_config(self.current_epoch)
        lr = config["lr"]
        nframes = config["nframes"]
        objective = config["loss"]
        use_avg_loss = config["avg_loss"]
        mode = config["mode"]
        
        self.use_weighted_loss = True if nframes >= 5 else False
        
        q = random.randint(0, 3) if not self.single else self.quality

        # Set Optimizers
        opt = self.optimizers()
        opt._optimizer.param_groups[0]["lr"] = lr

        # Batch: [B, T, C, H, W]
        seq_len = batch.shape[1]
        frames = [image.squeeze(1) for image in batch.chunk(seq_len, 1)][:nframes]

        # I frame compression
        with torch.no_grad():
            # (x, q_in_ckpt=False, q_index=None):
            self.i_frame_model.eval()
            x_hat = self.i_frame_model(frames[0], q_in_ckpt=True, q_index=q)["x_hat"]
            dpb = {
                "ref_frame": x_hat,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None,
                "ref_mv_y": None,
            }
            
            if batch_idx % 100 == 0:
                self.log_images(
                    {
                        f"train_x_ori_{0}": frames[0],
                        f"train_x_recon_{0}": dpb['ref_frame']
                    },
                    batch_idx,
                )

        # Iterative Update
        if mode == "inter":
            step = self.p_frame_model.forward_inter
        elif mode == "recon":
            step = self.p_frame_model.forward_recon
        elif mode == "all":
            step = self.p_frame_model.forward_all
        else:
            raise NotImplementedError

        total_psnr = AverageMeter()
        total_bpp = AverageMeter()
        total_mse = AverageMeter()
        total_loss = AverageMeter()

        avg_loss = 0
        for i in range(nframes - 1):
            # (x, dpb, q_index, frame_idx):
            out_net = step(frames[i + 1], dpb, q_index=q, frame_idx=i)
            dpb = out_net["dpb"]
            
            out_criterion = self.rate_distortion_loss(
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
            else:
                avg_loss += out_criterion["loss"]



            if batch_idx % 100 == 0:
                self.log_images(
                    {
                        f"train_x_ori_{i+1}": frames[i+1],
                        f"train_x_recon_{i+1}": dpb['ref_frame']
                    },
                    batch_idx,
                )

            total_psnr.update(out_criterion["psnr"].item())
            total_bpp.update(out_criterion["bpp"].item())
            total_mse.update(out_criterion["mse"].item())
            total_loss.update(out_criterion["loss"].item())

        if use_avg_loss:
            # TODO: should we divide avg_loss by sequence length? -> AdamW optimizer can deal with the avg_loss, but Adam optimizer can not without division. 
            opt.zero_grad()
            self.manual_backward(avg_loss / (nframes - 1))
            self.clip_gradients(
                opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt.step()

        self.log_dict(
            {
                "avg_psnr": total_psnr.avg,
                "avg_bpp": total_bpp.avg,
                "avg_mse": total_mse.avg,
                "avg_loss": total_loss.avg,
            },
            sync_dist=True,
        )


    def log_images(self, log_dict, batch_idx):
        if self.global_rank == 0:
            for key in log_dict.keys():
                self.logger.experiment.add_image(
                    key,
                    torchvision.utils.make_grid(torch.Tensor.cpu(log_dict[key])),
                    self.current_epoch * 100000 + batch_idx,
                    dataformats="CHW",
                )

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            nframes = 5
            objective = ["mv_rate", "x_rate", "x_dist"]
            self.use_weighted_loss = True if nframes >= 5 else False

            for q in range(4):
                # Set Optimizers
                # Batch: [B, T, C, H, W]
                seq_len = batch.shape[1]
                frames = [image.squeeze(1) for image in batch.chunk(seq_len, 1)]
                recon_frames = []

                # I frame compression
                # (x, q_in_ckpt=False, q_index=None):
                x_hat = self.i_frame_model(frames[0], q_in_ckpt=True, q_index=q)[
                    "x_hat"
                ]
                dpb = {
                    "ref_frame": x_hat,
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frames.append(x_hat)

                # Iterative Update
                step = self.p_frame_model.forward_all

                total_psnr = AverageMeter()
                total_bpp = AverageMeter()
                total_mse = AverageMeter()
                total_loss = AverageMeter()

                for i in range(nframes - 1):
                    # (x, dpb, q_index, frame_idx):
                    out_net = step(frames[i + 1], dpb, q_index=q, frame_idx=(i))
                    out_criterion = self.rate_distortion_loss(
                        out_net,
                        frames[i + 1],
                        q_index=q,
                        objective=objective,
                        frame_idx=i,
                    )

                    dpb = out_net["dpb"]
                    recon_frames.append(dpb["ref_frame"])

                    total_psnr.update(out_criterion["psnr"].item())
                    total_bpp.update(out_criterion["bpp"].item())
                    total_mse.update(out_criterion["mse"].item())
                    total_loss.update(out_criterion["loss"].item())

                self.log_dict(
                    {
                        f"val_avg_psnr/q{q}": total_psnr.avg,
                        f"val_avg_bpp/q{q}": total_bpp.avg,
                        f"val_avg_mse/q{q}": total_mse.avg,
                        f"val_avg_loss/q{q}": total_loss.avg,
                    },
                    sync_dist=True,
                )

                if batch_idx == 2:
                    self.log_images(
                        {
                            f"val_x_ori/q{q}": torch.cat(frames, dim=0),
                            f"val_x_recon/q{q}": torch.cat(recon_frames, dim=0),
                        },
                        batch_idx
                    )

    def configure_optimizers(self):
        parameters = {n for n, p in self.p_frame_model.named_parameters()}
        params_dict = dict(self.p_frame_model.named_parameters())

        optimizer = optim.AdamW(
            (params_dict[n] for n in sorted(parameters)),
            lr=1e-4,  # default
        )
        # optimizer = optim.Adam(
        #     (params_dict[n] for n in sorted(parameters)),
        #     lr=1e-4,  # default
        # )

        return {
            "optimizer": optimizer,
        }

    def load_i_frame_model(self):
        i_frame_net = IntraNoAR()
        ckpt = torch.load(
            "../../checkpoints/cvpr2023_image_psnr.pth.tar",
            map_location=torch.device("cpu"),
        )
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if "net" in ckpt:
            ckpt = ckpt["net"]
        consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")

        i_frame_net.load_state_dict(ckpt)
        i_frame_net.eval()
        return i_frame_net
