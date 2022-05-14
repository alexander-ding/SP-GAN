from pathlib import Path
import numpy as np
import torch

from Generation.config import parser
from Generation.model_test import ModelComfort

parser.add_argument("-i", type=str)
parser.add_argument("-o", type=str)
parser.add_argument("-s", type=str, choices=['male', 'female'])



def main(args):
    args.pretrain_model_G = "Chair_G.pth"
    args.log_dir = "models"

    model = ModelComfort(args)
    
    beta_files = sorted(Path(args.i).glob('*.pt'), key=lambda p: int(p.stem))
    betas = torch.stack([torch.load(f) for f in beta_files])
    
    sex = torch.zeros((len(betas), 2))
    sex[:, 0 if args.s == 'male' else 1] = 1.0
    betas = torch.cat((betas, sex), dim=-1)
    
    pcd = model.generate_conditioned(betas)
    print(pcd.shape)
    print(np.mean(np.min(pcd, axis=1), axis=0))
    print(np.mean(np.max(pcd, axis=1), axis=0))
    print(np.mean(np.prod(np.max(pcd, axis=1) - np.min(pcd, axis=1), axis=1)))

    OUT_DIR = Path(args.o)
    (OUT_DIR / "pcd").mkdir(exist_ok=True, parents=True)
    for i, f in enumerate(beta_files):
        np.save(OUT_DIR / "pcd" / f"{f.stem}.npy", pcd[i])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
