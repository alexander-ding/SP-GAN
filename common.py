import glob
import gzip
from pathlib import Path

import joblib
import numpy as np
import open3d as o3d
import torch

CUBOID = "CUBOID"
IM = "IM"
SPGAN = "mesh"

model_type = SPGAN

device = torch.device("cuda")

b_mean, b_std = torch.load("betas_params.pt")
default = torch.load('default_betas.pt')

def get_pose_batch(batch_size=1, ctype=model_type, cluster=-1):
    if ctype == CUBOID:
        bd_samples = "bd_samples"
    elif ctype == IM:
        bd_samples = "bd_samples_im"
    elif ctype == SPGAN:
        bd_samples = "body_pts"

    fs = []
    for i in range(batch_size):
        # Pick Cluster
        r = cluster if cluster is not -1 else np.random.randint(len(clustering))
        b = clustering[r]
        # Pick index
        i = np.random.choice(len(b))
        x = b[i]
        # File
        f = files[x]
        fs.append(f)

    pose_vecs = []
    pose_pts = []
    for f in fs:
        name = Path(f).stem

        pose_vec = torch.load(f)["pose"]

        n = DATA_DIR / bd_samples / f"{name}.gz"
        f = gzip.GzipFile(n, "r")
        pose_points = np.load(f)

        if not torch.is_tensor(pose_vec):
            pose_vec = torch.tensor(pose_vec)
        if not torch.is_tensor(pose_points):
            pose_points = torch.tensor(pose_points)

        pose_vecs.append(pose_vec)
        pose_pts.append(pose_points)

    pose_vecs = torch.stack(pose_vecs)
    pose_points = torch.stack(pose_pts)
    pose_encs = torch.tensor(encode_poses(pose_vecs.numpy().reshape(batch_size, -1)))

    return pose_vecs, pose_encs, pose_points

DATA_DIR = Path('/mnt/hdd1/chairs')
if model_type == CUBOID:
    files = sorted(glob.glob('data/chair')+glob.glob('data/gen_sample'))
elif model_type == IM:
    files = sorted(glob.glob('train_data'))
elif model_type == SPGAN:
    files = [DATA_DIR / "poses" / f"{i}.pt" for i in range(10000)]

def sample_betas(batch_size=1, is_default=False, return_z=False):
    if is_default:
        return default.repeat(batch_size, 1)
    z = torch.randn((batch_size, b_std.shape[-1])).float()
    betas = (b_mean + z * b_std).to(device).float()
    z = z.to(device)
    sex = torch.zeros((batch_size, 2)).to(device).float()
    sex[np.arange(batch_size), np.random.randint(0, 2, batch_size)] = 1.0
    z = torch.cat((z, sex), dim=-1)
    betas = torch.cat((betas, sex), dim=-1)
    
    if return_z:
        return betas, z
    else:
        return betas

def encode_betas(betas):
    betas[:,:16] = (betas[:,:16] - b_mean.to(device)) / b_std.to(device)
    return betas
def decode_betas(betas):
    betas[:,:16] = betas[:,:16] * b_std.to(device) + b_mean.to(device)
    return betas

BETAS_DIM = 18

clustering, clustering_inv = torch.load("clustered_poses.pt")
pca = joblib.load("pose_pca.joblib")

def encode_poses(poses):
    return pca.transform(poses)
def decode_poses(poses):
    return pca.inverse_transform(poses)

POSE_DIM = pca.n_components_

rfc = joblib.load("rfc.joblib")

def is_valid_pose(poses):
    return rfc.predict(poses)
