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
import open3d as o3d
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common import POSE_DIM, get_pose_batch, encode_poses, sample_betas, encode_betas, BETAS_DIM
from pc_encoder import HyperNetwork
from prox.proxy_models import get_comfort_loss, get_pose_loss
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


SAVE_DIR_POSE = Path("/mnt/hdd1/chairs/spgan-pose/")
SAVE_DIR_COMFORT = Path("/mnt/hdd1/chairs/spgan-comfort/")


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

    def condition_z(self, z, betas=None, return_betas=False):
        bs = len(z)
        if betas is None:
            _, betas = sample_betas(bs, return_z=True)

        z = z.cuda() * (1 / self.opts.nv)
        z = self.z_changer(betas, z.squeeze(1)).unsqueeze(1)
        z = z * self.opts.nv

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

        self.z_changer = HyperNetwork(BETAS_DIM, self.opts.nz)
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
            out_dir.mkdir(parents=True, exist_ok=True)

            n_batches = 100 // self.opts.bs

            for i in range(n_batches):
                x = self.sphere_generator(bs=self.opts.bs)
                z = self.noise_generator(bs=self.opts.bs)
            
                z_cond, betas = self.condition_z(z, return_betas=True)
                out_pc = self.G(x, z_cond)
                sample_pcs = transform_chair_pc(out_pc).cpu().detach().numpy()
                for j in range(self.opts.bs):
                    torch.save(
                        {
                            "betas": betas[j],
                            "pcd": sample_pcs[j],
                        },
                        out_dir / f"{epoch}_{i * self.opts.bs + j}.pt",
                    )

    def train_epoch(self, x):
        comfort_loss_avg = AverageValueMeter()
        for i in range(4000):
            z = self.noise_generator(bs=self.opts.bs)
            
            z_cond, betas = self.condition_z(z, return_betas=True)
            if i == 0:
                print("Input:", torch.mean(torch.var(z, axis=0)))
                print("Output:", torch.mean(torch.var(z_cond, axis=0)))

            out_pc = self.G(x, z_cond)
            out_pc = transform_chair_pc(out_pc)
            comfort_loss = get_comfort_loss(out_pc, betas, is_pcd=True) / self.opts.bs
            
            comfort_loss_avg.update(comfort_loss.detach())
            
            l_p = -self.normal.log_prob(z_cond)[torch.abs(z_cond) > (3 * 0.2)].sum() / torch.numel(z_cond)
            loss = comfort_loss + l_p
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        print(f"Comfort Loss: {comfort_loss_avg.avg}")
        return comfort_loss_avg.avg

    def train(self, start_epoch=0, end_epoch=100):
        self.normal = torch.distributions.Normal(0.0, self.opts.nv, validate_args=False)

        x = self.sphere_generator(bs=self.opts.bs)
        self.optim = optim.Adam(
            self.z_changer.parameters(), lr=1e-4
        )  # , weight_decay=1e-3)
        

        if start_epoch == 0:
            print(f"Visualizing: Epoch 0")
            self.visualize(SAVE_DIR_COMFORT / "raw", 0)
        
        for epoch in tqdm(range(start_epoch+1, end_epoch+1)):
            self.train_epoch(x)
            if epoch % 10 == 0:
                print(f"Visualizing: Epoch {epoch}")
                self.visualize(SAVE_DIR_COMFORT / "raw", epoch)
                torch.save(
                    self.z_changer.state_dict(), SAVE_DIR_COMFORT / f"z_changer_{epoch}.pt"
                )
                
    def generate_conditioned(self, betas):
        out = []
        for i in range(0, len(betas), self.opts.bs):
            bs = min(len(betas) - i, self.opts.bs)
            betas_batch = torch.as_tensor(encode_betas(np.asarray(betas[i:i+bs]))).cuda()
            x = self.sphere_generator(bs=bs)
            z = self.noise_generator(bs=bs)
            z_cond = self.condition_z(z, betas=betas_batch, return_betas=False)
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
            out_dir.mkdir(parents=True, exist_ok=True)

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
        self.visualize(SAVE_DIR_POSE / "raw", 0)

        for epoch in tqdm(range(1, epochs + 1)):
            self.train_epoch(x)
            if epoch % 10 == 0:
                print(f"Visualizing: Epoch {epoch}")
                self.visualize(SAVE_DIR_POSE / "raw", epoch)
                torch.save(
                    self.z_changer.state_dict(), SAVE_DIR_POSE / f"z_changer_{epoch}.pt"
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
