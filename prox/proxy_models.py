import json
import os

import numpy as np
import pc_encoder
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda")


def get_vf(cubes):
    verts = torch.tensor([], dtype=torch.float)
    faces = torch.tensor([], dtype=torch.long)
    for cube in cubes:
        v, f = cube.getTris()
        if v is not None and f is not None:
            faces = torch.cat((faces, (f + verts.shape[0])))
            verts = torch.cat((verts, v))
    return verts.to(device), faces.to(device)


def sample_mesh_surface(v, f, n_s=10000):

    a, b, c = v[f].permute(1, 0, 2)
    areas = torch.cross(a - b, b - c).norm(dim=-1)
    weights = (areas / areas.sum()).detach().cpu().numpy()

    choices = np.random.choice(a=len(weights), size=n_s, p=weights)
    u, v = torch.rand(size=(2, n_s)).to(device)

    pts = (1 - u ** 0.5).view(-1, 1) * a[choices]
    pts += (u ** 0.5 * (1 - v)).view(-1, 1) * b[choices]
    pts += (v * u ** 0.5).view(-1, 1) * c[choices]

    return pts


class LossProxy(nn.Module):
    def __init__(self):
        super(LossProxy, self).__init__()

        reshape_sz = 1024
        dropout = 0.2

        self.encoder = pc_encoder.PCEncoder(beta_sz=16+2)

        linear1 = pc_encoder.FeedForward(1024,256,1,['leaky_relu']*2+['linear'])
        
        self.loss_network = nn.Sequential(
            linear1,
            nn.Flatten(0)
        )

    def predict(self,inputs,betas):

        encoding = self.encoder(inputs,betas)
        output = self.loss_network(encoding)
        return output

    def forward(self,chairs,betas,is_pcd):
        
        pts_chair = []
        if not is_pcd:
            for chair in chairs:
                v,f = get_vf(chair)
                p = sample_mesh_surface(v.unsqueeze(0),f).squeeze(0)
                pts_chair.append(p)
            chairs = torch.stack(pts_chair)

        return self.predict(chairs,betas).flatten()

    def loss(self,pred,labels):

        loss = 0.
        loss += (pred - labels).abs().sum()

        return loss

    def contrastive(self,pred,labels):

        num_choices = max(len(pred),5000)
        choices = np.random.choice(len(pred),(2*num_choices))

        p1,p2 = pred[choices].view(2,-1)
        t1,t2 = labels[choices].view(2,-1)

        truth = t1 < t2
        p_comp = (p1 < p2) == truth
            
        return p_comp.float().mean()


class PoseProxy(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super(PoseProxy, self).__init__()

        reshape_sz = 256

        hsz0 = reshape_sz // 2
        hsz1 = reshape_sz // 4
        hsz2 = reshape_sz // 8

        dropout = 0.4

        self.encoder = pc_encoder.PCEncoderPose()

        self.linear1 = nn.Linear(reshape_sz, 1)

        self.loss_network = nn.Sequential(self.linear1, nn.Flatten(0))

        self.mean = mean
        self.std = std

    def predict(self, inputs):

        encoding = self.encoder(inputs)
        return self.loss_network(encoding)

    def forward(self, chairs, pose, is_pcd=False):

        pts_chair = []
        if not is_pcd:
            for chair in chairs:
                v, f = get_vf(chair)
                p = sample_mesh_surface(v.unsqueeze(0), f).squeeze(0)
                pts_chair.append(p)
            chairs = torch.stack(pts_chair)

        cshape = (chairs.shape[0], chairs.shape[1], 1)
        pshape = (pose.shape[0], pose.shape[1], 1)

        chairs = torch.cat(
            (chairs, torch.ones(cshape).to(device), torch.zeros(cshape).to(device)),
            dim=-1,
        )
        pose = torch.cat(
            (pose, torch.zeros(pshape).to(device), torch.ones(pshape).to(device)),
            dim=-1,
        )

        pts = torch.cat((chairs, pose), dim=1)

        return self.predict(pts).flatten()

    def loss(self, pred, labels):

        labels = (labels - self.mean) / self.std

        loss = 0.0
        loss += (pred - labels).abs().sum()

        return loss

    def recognition(self, pred):

        p_truth, p_random = pred.view(-1, 2).T

        return (p_truth < p_random).float().mean()

    def contrastive(self, pred, labels):

        labels = (labels - self.mean) / self.std
        num_choices = max(len(pred), 5000)
        choices = np.random.choice(len(pred), (2 * num_choices))

        p1, p2 = pred[choices].view(2, -1)
        t1, t2 = labels[choices].view(2, -1)

        truth = t1 < t2
        p_comp = (p1 < p2) == truth

        return p_comp.float().mean()


# loss_model = LossProxy().to(device)
# loss_model.load_state_dict(torch.load("comf_dict_mesh.pt"))
# for parameter in loss_model.parameters():
#     parameter.requires_grad = False
# loss_model.eval()


# def get_comfort_loss(inputs, betas, is_pcd=False):
#     return loss_model(inputs, betas, is_pcd).sum()


loss_model = LossProxy().to(device)
loss_model.load_state_dict(torch.load("comf_dict_cluster_mesh.pt"))
for parameter in loss_model.parameters():
    parameter.requires_grad = False
loss_model.eval()
    
def get_comfort_loss(cubes,betas,is_pcd=False):
    return loss_model(cubes,betas,is_pcd).sum()

pose_model = PoseProxy().to(device)
pose_model.load_state_dict(torch.load("pose_dict_cluster_mesh.pt"))
for parameter in pose_model.parameters():
    parameter.requires_grad = False
pose_model.eval()


def get_pose_loss(chairs, pose, is_pcd=False):
    return pose_model(chairs, pose, is_pcd).sum()

