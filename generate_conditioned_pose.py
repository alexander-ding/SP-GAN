from pathlib import Path
import numpy as np
import torch

from Generation.config import parser
from Generation.model_test import ModelPose

parser.add_argument("-i", type=str)
parser.add_argument("-o", type=str)


def main(args):
    args.pretrain_model_G = "Chair_G.pth"
    args.log_dir = "models"

    model = ModelPose(args)
    
    pose_files = sorted(Path(args.i).glob('*.pt'), key=lambda p: int(p.stem))
    poses = torch.stack([torch.load(f) for f in pose_files])
    
    pcd = model.generate_conditioned(poses)
    print(pcd.shape)
    print(np.mean(np.min(pcd, axis=1), axis=0))
    print(np.mean(np.max(pcd, axis=1), axis=0))
    print(np.mean(np.prod(np.max(pcd, axis=1) - np.min(pcd, axis=1), axis=1)))

    OUT_DIR = Path(args.o)
    (OUT_DIR / "pcd").mkdir(exist_ok=True, parents=True)
    for i, f in enumerate(pose_files):
        np.save(OUT_DIR / "pcd" / f"{f.stem}.npy", pcd[i])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
