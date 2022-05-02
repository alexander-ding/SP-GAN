# encoding=utf-8

import gzip
import logging
import math
import os
import random
import sys
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common import POSE_DIM, get_pose_batch, encode_poses
from pc_encoder import HyperNetwork
from prox.proxy_models import get_pose_loss
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from Generation.Generator import Generator

cudnn.benchnark = True


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


SAVE_DIR = Path("/mnt/hdd1/chairs/spgan-pose/")


def pc_normalize(pc, return_len=False):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    if return_len:
        return m
    return pc


def transform_chair_pc(x):
    x = x.transpose(2, 1)
    centroid = torch.mean(x, dim=1, keepdims=True)
    x = x - centroid
    furthest_distance = torch.amax(
        torch.sqrt(torch.sum(x ** 2, dim=-1, keepdims=True)), dim=1, keepdims=True
    )
    x = x / furthest_distance
    x = torch.stack([-x[:, :, 0], x[:, :, 1], -x[:, :, 2]], dim=2) * 1.18
    return x


class ModelVanilla(object):
    def __init__(self, opts):
        self.opts = opts
        self.build_model_eval()
        could_load, _ = self.load(self.opts.log_dir)
        self.betas_mean, self.betas_std = torch.load("betas_params.pt")
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

    def build_model_eval(self):
        """Models"""
        self.G = Generator(self.opts)
        self.ball = None
        print(
            "# generator parameters:",
            sum(param.numel() for param in self.G.parameters()),
        )

        self.G.cuda()
        for param in self.G.parameters():
            param.requires_grad = False

    def sample(self, z):
        x = self.sphere_generator(bs=len(z))
        out_pc = self.G(x, z)
        sample_pcs = transform_chair_pc(out_pc).detach().numpy()

        return sample_pcs

    def noise_generator(self, bs=1, masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(
                    0, self.opts.nv, (bs, self.opts.np, self.opts.nz)
                )
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))

            if self.opts.n_mix and random.random() < 0.8:
                noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
                for i in range(bs):
                    idx = np.arange(self.opts.np)
                    np.random.shuffle(idx)
                    num = int(random.random() * self.opts.np)
                    noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i, idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise

    def sphere_generator(self, bs=2, static=True):

        if self.ball is None:
            if static:
                self.ball = np.loadtxt("template/balls/2048.xyz")[:, :3]
            else:
                self.ball = np.loadtxt("template/balls/ball2.xyz")[:, :3]
            self.ball = pc_normalize(self.ball)

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball)).cuda()

        return ball

    def read_ball(self, sort=False):
        x = np.loadtxt("template/balls/2048.xyz")
        ball = pc_normalize(x)

        N = ball.shape[0]
        # xx = torch.bmm(x, x.transpose(2,1))
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]

        # order = np.argsort(dist[1000])[::1]
        # ball = ball[order]

        return ball

    def simple_gen(self, number=50, betas=None):
        x = self.sphere_generator(self.opts.bs)
        all_sample = []
        zs = []
        n_batches = number // self.opts.bs + 1
        for _ in range(n_batches):
            with torch.no_grad():
                z = self.noise_generator(bs=self.opts.bs)
                out_pc = self.G(x, z)
            zs.append(z)
            all_sample.append(out_pc)

        sample_pcs = torch.cat(all_sample, dim=0)[:number]
        sample_pcs = transform_chair_pc(sample_pcs).cpu().detach().numpy()
        zs = torch.cat(zs, dim=0)[:number].cpu().detach().numpy()
        return sample_pcs, zs

    def load(self, checkpoint_dir):
        if self.opts.pretrain_model_G is None:
            print("################ new training ################")
            return False, 1

        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        # ----------------- load G -------------------
        if not self.opts.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.opts.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G)
            if flag_G == False:
                print("G--> Error: no checkpoint directory found!")
                exit()
            else:
                print("resume_file_G------>: {}".format(resume_file_G))
                checkpoint = torch.load(resume_file_G)
                self.G.load_state_dict(checkpoint["G_model"])
                # self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint["G_epoch"]
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        print(
            " [*] Success to load model --> {} & {}".format(
                self.opts.pretrain_model_G, self.opts.pretrain_model_z
            )
        )
        return True, G_epoch


class ModelComfort(object):
    def __init__(self, opts):
        self.opts = opts
        self.build_model_eval()
        could_load, _ = self.load(self.opts.log_dir)
        self.betas_mean, self.betas_std = torch.load("betas_params.pt")
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

    def sample(self, zs):
        x = self.sphere_generator(bs=len(zs))
        betas = self.sample_betas(bs=len(zs))
        z_cond = self.condition_z(zs, betas=betas)
        out_pc = self.G(x, z_cond)
        sample_pcs = transform_chair_pc(out_pc)
        return sample_pcs, betas

    def sample_betas(self, bs, sex=None):
        sex_dict = {"male": 0, "female": 1}
        sex = np.random.randint(0, 2, size=bs) if sex is None else sex_dict[sex]
        s_oh = torch.zeros(bs, 2)
        s_oh[np.arange(bs), sex] = 1.0
        betas = (
            torch.randn((bs, self.betas_mean.size()[-1])) * self.betas_std
            + self.betas_mean
        ).float()
        return torch.cat((betas, s_oh), dim=1)

    def condition_z(self, z, betas=None, return_betas=False):
        bs = len(z)
        if betas is None:
            betas = self.sample_betas(bs)
        betas = betas.cuda()
        z = z.cuda()
        z = self.z_changer(betas, z.squeeze(1))
        z = z.unsqueeze(1)
        if return_betas:
            return z, betas
        else:
            return z

    def build_model_eval(self):
        """Models"""
        self.G = Generator(self.opts)
        self.ball = None
        print(
            "# generator parameters:",
            sum(param.numel() for param in self.G.parameters()),
        )

        self.G.cuda()
        for param in self.G.parameters():
            param.requires_grad = False

        self.z_changer = MLP_FiLM(18, self.opts.nz)
        self.z_changer.cuda()

    def noise_generator(self, bs=1, masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(
                    0, self.opts.nv, (bs, self.opts.np, self.opts.nz)
                )
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))
                # scale = self.opts.nv
                # w = np.random.uniform(low=-scale, high=scale, size=(bs, 1, self.opts.nz))
                noise = np.tile(noise, (1, self.opts.np, 1))

            if self.opts.n_mix and random.random() < 0.8:
                noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
                for i in range(bs):
                    idx = np.arange(self.opts.np)
                    np.random.shuffle(idx)
                    num = int(random.random() * self.opts.np)
                    noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i, idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise

    def sphere_generator(self, bs=2, static=True):

        if self.ball is None:
            if static:
                self.ball = np.loadtxt("template/balls/10000.xyz")[:, :3]
            else:
                self.ball = np.loadtxt("template/balls/ball2.xyz")[:, :3]
            self.ball = pc_normalize(self.ball)

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball)).cuda()

        return ball

    def read_ball(self, sort=False):
        x = np.loadtxt("template/balls/2048.xyz")
        ball = pc_normalize(x)

        N = ball.shape[0]
        # xx = torch.bmm(x, x.transpose(2,1))
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]

        # order = np.argsort(dist[1000])[::1]
        # ball = ball[order]

        return ball

    def simple_gen(self, number=50, betas=None):
        x = self.sphere_generator(self.opts.bs)
        all_sample = []
        n_batches = number // self.opts.bs + 1
        for _ in range(n_batches):
            with torch.no_grad():
                z = self.noise_generator(bs=self.opts.bs)
                z_cond = self.condition_z(z, betas=betas)
                out_pc = self.G(x, z_cond)

            all_sample.append(out_pc)

        sample_pcs = torch.cat(all_sample, dim=0)[:number]
        sample_pcs = transform_chair_pc(sample_pcs).detach().numpy()

        return sample_pcs

    def visualize(self, out_dir):
        with torch.no_grad():
            if not out_dir.exists():
                out_dir.mkdir()

            for i in range(10):
                x = self.sphere_generator(bs=10)
                sex = "male" if i < 5 else "female"
                betas = self.sample_betas(10, sex=sex)
                zs = self.noise_generator(bs=10)
                z_cond = self.condition_z(zs, betas=betas)
                out_pc = self.G(x, z_cond)
                sample_pcs = transform_chair_pc(out_pc).cpu().detach().numpy()
                for j in range(10):
                    torch.save(
                        {
                            "betas": betas[j],
                            "pcd": sample_pcs[j],
                        },
                        out_dir / f"{i * 10 + j}.pt",
                    )

    def train(self, epochs=100):
        writer = SummaryWriter(logdir=SAVE_DIR / "z_changer")

        self.optim = optim.Adam(self.z_changer.parameters(), lr=1e-5)
        c_disc_loss = 10.0
        x = self.sphere_generator(bs=self.opts.bs)

        comfort_loss_avg = AverageValueMeter()
        disc_loss_avg = AverageValueMeter()
        gen_loss_avg = AverageValueMeter()

        for epoch in tqdm(range(1, epochs + 1)):
            for i in range(1000):
                criterion = nn.BCELoss()

                z = self.noise_generator(bs=self.opts.bs)
                z_cond, betas = self.condition_z(z, return_betas=True)
                p_gen = self.discriminator(z_cond[:, 0, :])
                gen_loss = criterion(p_gen, torch.zeros_like(p_gen))
                gen_loss *= c_disc_loss

                out_pc = self.G(x, z_cond)
                out_pc = transform_chair_pc(out_pc)
                comfort_loss = get_comfort_loss(out_pc, betas, is_pcd=True)

                c_loss = 0.0
                c_loss += comfort_loss
                c_loss += gen_loss

                self.comf_optim.zero_grad()
                c_loss.backward()
                self.comf_optim.step()
                comfort_loss_avg.update(comfort_loss.detach().cpu().numpy())
                gen_loss_avg.update(gen_loss.detach().cpu().numpy())

                z = self.noise_generator(bs=self.opts.bs)
                z_cond, betas = self.condition_z(z, return_betas=True)
                p_gen = self.discriminator(z_cond[:, 0, :])
                p_real = self.discriminator(z[:, 0, :])
                disc_loss = criterion(p_gen, torch.ones_like(p_gen)) + criterion(
                    p_real, torch.zeros_like(p_gen)
                )
                disc_loss *= c_disc_loss

                d_loss = 0.0
                d_loss += disc_loss

                self.disc_optim.zero_grad()
                d_loss.backward()
                self.disc_optim.step()
                disc_loss_avg.update(d_loss.detach().cpu().numpy())

            print(
                f"Comfort Loss: {comfort_loss_avg.avg:.3f} Gen Loss: {gen_loss_avg.avg:.3f} Discriminator Loss: {disc_loss_avg.avg:.3f}"
            )
            if epoch % 25 == 0:
                print(f"Visualizing: Epoch {epoch}")
                self.visualize(SAVE_DIR / f"{epoch}")
                torch.save(
                    self.z_changer.state_dict(), SAVE_DIR / f"z_changer_{epoch}.pt"
                )

    def load(self, checkpoint_dir):
        if self.opts.pretrain_model_G is None:
            print("################ new training ################")
            return False, 1

        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        # ----------------- load G -------------------
        if not self.opts.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.opts.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G)
            if flag_G == False:
                print("G--> Error: no checkpoint directory found!")
                exit()
            else:
                print("resume_file_G------>: {}".format(resume_file_G))
                checkpoint = torch.load(resume_file_G)
                self.G.load_state_dict(checkpoint["G_model"])
                # self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint["G_epoch"]
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load Z -------------------
        if not self.opts.pretrain_model_z is None:
            resume_file_z = os.path.join(checkpoint_dir, self.opts.pretrain_model_z)
            flag_z = os.path.isfile(resume_file_z)
            if flag_z == False:
                print("Z--> Error: no checkpoint directory found!")
                exit()
            else:
                print("resume_file_z------>: {}".format(resume_file_z))
                checkpoint = torch.load(resume_file_z)
                self.z_changer.load_state_dict(checkpoint)
        else:
            print(" [*] Failed to find the pretrain_model_z")

        print(
            " [*] Success to load model --> {} & {}".format(
                self.opts.pretrain_model_G, self.opts.pretrain_model_z
            )
        )
        return True, G_epoch


class ModelPose(object):
    def __init__(self, opts):
        self.opts = opts
        self.build_model_eval()
        could_load, _ = self.load(self.opts.log_dir)
        self.betas_mean, self.betas_std = torch.load("betas_params.pt")
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0) 

    def condition_z(self, z, pose=None, return_pose=False):
        bs = len(z)
        if pose is None:
            sampled = get_pose_batch(bs)
            true_pose, encoded_pose, pose_pcd = (
                sampled[0].cuda(),
                sampled[1].cuda(),
                sampled[2].cuda(),
            )
        else:
            encoded_pose = pose
        
        z = z.cuda() * (1 / self.opts.nv)
        z = self.z_changer(encoded_pose, z.squeeze(1)).unsqueeze(1)
        z = z * self.opts.nv

        if return_pose:
            return z, pose_pcd, true_pose
        else:
            return z

    def build_model_eval(self):
        """Models"""
        self.G = Generator(self.opts)
        self.ball = None
        print(
            "# generator parameters:",
            sum(param.numel() for param in self.G.parameters()),
        )

        self.G.cuda()
        for param in self.G.parameters():
            param.requires_grad = False

        self.z_changer = HyperNetwork(POSE_DIM, self.opts.nz)
        self.z_changer.cuda()

    def noise_generator(self, bs=1, masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(
                    0, self.opts.nv, (bs, self.opts.np, self.opts.nz)
                )
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))

            if self.opts.n_mix and random.random() < 0.8:
                noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
                for i in range(bs):
                    idx = np.arange(self.opts.np)
                    np.random.shuffle(idx)
                    num = int(random.random() * self.opts.np)
                    noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i, idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise

    def sphere_generator(self, bs=2, static=True):

        if self.ball is None:
            if static:
                self.ball = np.loadtxt("template/balls/2048.xyz")[:, :3]
            else:
                self.ball = np.loadtxt("template/balls/ball2.xyz")[:, :3]
            self.ball = pc_normalize(self.ball)

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball)).cuda()

        return ball

    def read_ball(self, sort=False):
        x = np.loadtxt("template/balls/2048.xyz")
        ball = pc_normalize(x)

        N = ball.shape[0]
        # xx = torch.bmm(x, x.transpose(2,1))
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]

        # order = np.argsort(dist[1000])[::1]
        # ball = ball[order]

        return ball

    def visualize(self, out_dir, epoch):
        with torch.no_grad():
            if not out_dir.exists():
                out_dir.mkdir()

            n_batches = 100 // self.opts.bs

            for i in range(n_batches):
                x = self.sphere_generator(bs=self.opts.bs)
                sex = "male" if i < 5 else "female"
                z = self.noise_generator(bs=self.opts.bs)
                z_cond, pose_pts, pose_vec = self.condition_z(z, return_pose=True)
                out_pc = self.G(x, z_cond)
                sample_pcs = transform_chair_pc(out_pc).cpu().detach().numpy()
                for j in range(self.opts.bs):
                    torch.save(
                        {
                            "pose": pose_vec[j],
                            "pose_pts": pose_pts[j],
                            "pcd": sample_pcs[j],
                        },
                        out_dir / f"{epoch}_{i * self.opts.bs + j}.pt",
                    )

    def train_epoch(self, x):
        pose_loss_avg = AverageValueMeter()
        sample_pcs = None
        pose_pts = None
        for i in range(2000):
            z = self.noise_generator(bs=self.opts.bs)
            
            z_cond, pose_pts, pose_vec = self.condition_z(z, return_pose=True)
            if i == 0:
                print("Input:", torch.mean(torch.var(z, axis=0)))
                print("Output:", torch.mean(torch.var(z_cond, axis=0)))

            out_pc = self.G(x, z_cond)
            out_pc = transform_chair_pc(out_pc)
            pose_loss = get_pose_loss(out_pc, pose_pts, is_pcd=True) / self.opts.bs
            
            pose_loss_avg.update(pose_loss.detach())
            
            l_p = -self.normal.log_prob(z_cond)[torch.abs(z_cond) > (3 * 0.2)].sum() / torch.numel(z_cond)
            loss = pose_loss + l_p
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            sample_pcs = out_pc.cpu().detach().numpy()
        sample_pcs = sample_pcs[0]
        pose_pts = pose_pts[0]
        original_chair = transform_chair_pc(self.G(x, z)).cpu().detach().numpy()[0]
        torch.save(
            {
                "pose": pose_pts,
                "new_chair": sample_pcs,
                "original_chair": original_chair,
            },
            "temp.pt",
        )
        print(f"Pose Loss: {pose_loss_avg.avg}")
        return pose_loss_avg.avg

    def train(self, epochs=100):
        self.normal = torch.distributions.Normal(0.0, self.opts.nv, validate_args=False)

        x = self.sphere_generator(bs=self.opts.bs)
        self.optim = optim.Adam(
            self.z_changer.parameters(), lr=1e-4
        )  # , weight_decay=1e-3)
        

        print(f"Visualizing: Epoch 0")
        self.visualize(SAVE_DIR / "pcd", 0)

        for epoch in tqdm(range(1, epochs + 1)):
            self.train_epoch(x)
            if epoch % 10 == 0:
                print(f"Visualizing: Epoch {epoch}")
                self.visualize(SAVE_DIR / "pcd", epoch)
                torch.save(
                    self.z_changer.state_dict(), SAVE_DIR / f"z_changer_{epoch}.pt"
                )
                
    def generate_conditioned(self, poses):
        out = []
        for i in range(0, len(poses), self.opts.bs):
            bs = min(len(poses) - i, self.opts.bs)
            pose_batch = torch.as_tensor(encode_poses(np.asarray(poses[i:i+bs]))).cuda()
            x = self.sphere_generator(bs=bs)
            z = self.noise_generator(bs=bs)
            z_cond = self.condition_z(z, pose=pose_batch, return_pose=False)
            out_pc = self.G(x, z_cond)
            out_pc = transform_chair_pc(out_pc)
            out.append(out_pc)
        return torch.cat(out).cpu().detach().numpy()
        

    def load(self, checkpoint_dir):
        if self.opts.pretrain_model_G is None:
            print("################ new training ################")
            return False, 1

        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        # ----------------- load G -------------------
        if not self.opts.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.opts.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G)
            if flag_G == False:
                print("G--> Error: no checkpoint directory found!")
                exit()
            else:
                print("resume_file_G------>: {}".format(resume_file_G))
                checkpoint = torch.load(resume_file_G)
                self.G.load_state_dict(checkpoint["G_model"])
                # self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint["G_epoch"]
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load Z -------------------
        if not self.opts.pretrain_model_z is None:
            resume_file_z = os.path.join(checkpoint_dir, self.opts.pretrain_model_z)
            flag_z = os.path.isfile(resume_file_z)
            if flag_G == False:
                print("G--> Error: no checkpoint directory found!")
                exit()
            else:
                print("resume_file_z------>: {}".format(resume_file_z))
                checkpoint = torch.load(resume_file_z)
                self.z_changer.load_state_dict(checkpoint)
        else:
            print(" [*] Failed to find the pretrain_model_z")

        print(
            " [*] Success to load model --> {} & {}".format(
                self.opts.pretrain_model_G, self.opts.pretrain_model_z
            )
        )
        return True, G_epoch
