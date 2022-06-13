import numpy as np
import torch
import random
import os

BASE_PATH  = '/kaggle/input/uw-madison-gi-tract-image-segmentation'

class CFG:

    def __init__(self,debug = True, exp_name = 'Baselinev2',
                comment = 'unet-efficientnet_b1-224x224-aug2-split2', model_name = 'Unet',
                backbone = 'efficientnet-b1', train_bs = 128,valid_bs = 256, epochs = 15,
                lr = 2e-3, scheduler = 'CosineAnnealingLR',
                min_lr = 1e-6, n_fold = 5,fold_no = 0) -> None:

        self.seed          = 101
        self.debug         = debug # set debug=False for Full Training
        self.exp_name      = exp_name
        self.comment       = comment
        self.model_name    = model_name
        self.backbone      = backbone
        self.train_bs      = train_bs
        self.valid_bs      = train_bs*2
        self.img_size      = [224, 224]
        self.epochs        = epochs
        self.lr            = lr
        self.scheduler     = scheduler
        self.min_lr        = min_lr
        self.T_max         = int(30000/train_bs*epochs)+50
        self.T_0           = 25
        self.warmup_epochs = 0
        self.wd            = 1e-6
        self.n_accumulate  = max(1, 32//train_bs)
        self.n_fold        = n_fold
        self.num_classes   = 3
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fold_no       = fold_no


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

#this line will go in main or doesnt have to if it prints seeding done.

def initialise_config(args):
    cfg = CFG(**args)
    set_seed(CFG.seed)
    return cfg

