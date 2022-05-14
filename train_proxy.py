#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import pprint
from datetime import datetime

from Generation.config import parser
from Generation.model_test import ModelComfort, ModelPose

pp = pprint.PrettyPrinter()

opts = parser.parse_args()


if __name__ == '__main__':
    opts.pretrain_model_G = "Chair_G.pth"
    opts.log_dir = "models"
    opts.np = 2048
    opts.bs = 10

    model = ModelComfort(opts)
    model.train(start_epoch=100, end_epoch=200)
