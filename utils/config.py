import numpy as np
import torch
import random
import os

BASE_PATH  = '/kaggle/input/uw-madison-gi-tract-image-segmentation'

class CFG:

    def __init__(self,args) -> None:

        self.seed          = args.seed
        self.debug         = args.debug # set debug=False for Full Training
        self.exp_name      = args.exp_name
        self.comment       = args.comment
        self.model_name    = args.model_name
        self.backbone      = args.backbone
        self.train_bs      = args.train_bs
        self.valid_bs      = self.train_bs*2
        self.img_size      = [224, 224]
        self.epochs        = args.epochs
        self.lr            = args.lr
        self.scheduler     = args.scheduler
        self.min_lr        = 1e-6
        self.T_max         = int(30000/self.train_bs*self.epochs)+50
        self.T_0           = 25
        self.warmup_epochs = 0
        self.wd            = 1e-6
        self.n_accumulate  = max(1, 32//self.train_bs)
        self.n_fold        = args.n_fold
        self.num_classes   = 3
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fold_no       = args.fold_no



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
    cfg = CFG(args)
    set_seed(cfg.seed)
    return cfg

