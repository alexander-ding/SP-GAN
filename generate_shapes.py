#!/usr/bin/env python
# -*- coding:utf-8 _*-
from Generation.config import opts
from Generation.model_test import ModelVanilla
from datetime import datetime
import os
import pprint
import torch
import numpy as np
from pathlib import Path
pp = pprint.PrettyPrinter()

# used on Feb 9 to generate 10,000 pointclouds
if __name__ == '__main__':

    opts.pretrain_model_G = "Chair_G.pth"
    opts.log_dir = "models"

    model = ModelVanilla(opts)
    pcd = model.simple_gen(10000)
    
    print(np.mean(np.min(pcd, axis=1), axis=0))
    print(np.mean(np.max(pcd, axis=1), axis=0))
    print(np.mean(np.prod(np.max(pcd, axis=1) - np.min(pcd, axis=1), axis=1)))
    
    OUT_DIR = Path('/mnt/hdd1/chairs/pcd')
    for i in range(10000):
        np.save(OUT_DIR / f'{i}.npy', pcd[i])
        