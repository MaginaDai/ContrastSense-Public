import torch
import sys
import os
from os.path import dirname
sys.path.append(dirname(sys.path[0]))
sys.path.append(dirname(dirname(sys.path[0])))

from baseline.GILE.model_GILE import GILE

def load_model(args):
    if args.now_model_name == 'GILE':
        model = GILE(args)
    else:
        print('model not available!\n')
    return model

def set_up_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer
