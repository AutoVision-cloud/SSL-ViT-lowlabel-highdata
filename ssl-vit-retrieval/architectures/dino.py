import architectures.vision_transformer as vits
from architectures.vision_transformer import DINOHead
import architectures.utils as utils
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = vits.__dict__['vit_small'](patch_size=16)
        utils.load_pretrained_weights(self.model, '', 'teacher', 'vit_small', 16)
        
        self.name = opt.arch
        self.model.last_linear = torch.nn.Linear(384, opt.embed_dim)

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

    def forward(self, x, **kwargs):
        x = self.model(x)
        enc_out = x = x.view(x.size(0),-1)
        
        x = self.model.last_linear(x)
        
        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x, (enc_out, enc_out.unsqueeze(-1).unsqueeze(-1))

