from Generation.config import opts
from Generation.model_test import ModelComfort, ModelPose, ModelVanilla, get_pose_batch
from pathlib import Path
import torch
opts.pretrain_model_G = "Chair_G.pth"
opts.log_dir = "models"
opts.pretrain_model_z = "z_changer_comfort.pt"
opts.np = 2048
import trimesh

def mkdir(p):
    if not p.exists():
        p.mkdir()

DATA_DIR = Path('/mnt/hdd1/chairs/comfort/examples')
mkdir(DATA_DIR)
mkdir(DATA_DIR / 'pcd')
# model_vanilla = ModelVanilla(opts)
model_comf = ModelComfort(opts)
for j in range(10):
    zs = model_comf.noise_generator(10)
    pcds, betas = model_comf.sample(zs)
    for i in range(10):
        output_i = j * 10 + i
        sex = 'male' if betas[i][16] == 1. else 'female'
        beta = betas[i][:16]
        torch.save({
            'pcd': pcds[i],
            'betas': beta,
            'sex': sex,
        }, DATA_DIR / 'pcd' / f'{output_i}.pt')