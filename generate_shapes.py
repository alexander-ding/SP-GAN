#!/usr/bin/env python
# -*- coding:utf-8 _*-
from pathlib import Path

import numpy as np

from Generation.config import parser
from Generation.model_test import ModelVanilla

parser.add_argument("-n", type=int)
parser.add_argument("-o", type=str)


def main(args):
    args.pretrain_model_G = "Chair_G.pth"
    args.log_dir = "models"

    model = ModelVanilla(args)
    pcd, zs = model.simple_gen(args.n)

    print(np.mean(np.min(pcd, axis=1), axis=0))
    print(np.mean(np.max(pcd, axis=1), axis=0))
    print(np.mean(np.prod(np.max(pcd, axis=1) - np.min(pcd, axis=1), axis=1)))

    OUT_DIR = Path(args.o)
    (OUT_DIR / "pcd").mkdir(exist_ok=True, parents=True)
    (OUT_DIR / "z").mkdir(exist_ok=True, parents=True)
    for i in range(args.n):
        np.save(OUT_DIR / "pcd" / f"{i}.npy", pcd[i])
        np.save(OUT_DIR / "z" / f"{i}.npy", zs[i])


# used on Feb 9 to generate 10,000 pointclouds
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
