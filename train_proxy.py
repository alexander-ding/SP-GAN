#!/usr/bin/env python
# -*- coding:utf-8 _*-
from Generation.config import opts
from Generation.model_test import ModelComfort, ModelPose
from datetime import datetime
import os
import pprint
pp = pprint.PrettyPrinter()


if __name__ == '__main__':
    opts.pretrain_model_G = "Chair_G.pth"
    opts.log_dir = "models"

    model = ModelPose(opts)
    model.train()
