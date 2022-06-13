import numpy as np
import gc
import torch
import torch.optim as optim
import argparse

from yaml import parse

from models.unet import build_model
from utils.config import CFG,initialise_config
from utils.loss_func import fetch_scheduler
from src.dataloader import prepare_loaders
from src.engine import run_training
from src.dataloader import create_folds,get_mask_paths

# parser.add_argument('--$(variable_name)', default=$(variable_default_value), type=$(variable_data_type))
def get_args_parser():
    parser = argparse.ArgumentParser('Set training/testing configuration', add_help=False)
    parser.add_argument('--debug', default=True, type=bool)
    parser.add_argument('--exp_name', default='UWMIS - Segmentation', type=str)
    parser.add_argument('--train_bs', default = 128, type= int)
    parser.add_argument('lr',default=2e-3,type=float)
    parser.add_argument('model_name',default="Unet",type=str)
    parser.add_argument('comment',default="",type=str)
    parser.add_argument('scheduler',default='CosineAnnealingLR',type=str)
    parser.add_argument('epochs',default=15,type=int)
    parser.add_argument('n_fold',default=5,type=int)
    parser.add_argument('backbone',default='efficientnet-b1',type=str)
    parser.add_argument('fold_no',default=0,type=int)
    
    return parser

def main(args):

    cfg = initialise_config(**args)
    model = build_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = fetch_scheduler(optimizer,cfg)

    for fold in range(cfg.fold_no,cfg.fold_no+1):
        print(f'#'*30)
        print(f'### Fold: {fold}')
        print(f'#'*30)
        # run = wandb.init(project='uw-maddison-gi-tract', 
        #                 config={k:v for k, v in dict(vars(CFG)).items() if '__' not in k},
        #                 anonymous=anonymous,
        #                 name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
        #                 group=CFG.comment,
        #                 )
        train_loader, valid_loader = prepare_loaders(fold=fold,
                                    df = create_folds(get_mask_paths(),cfg),
                                    debug=args.debug,
                                    cfg=cfg)

        model     = build_model(cfg=cfg)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        scheduler = fetch_scheduler(optimizer)
        model, history = run_training(model, optimizer, scheduler,
                                    device=cfg.device,
                                    num_epochs=cfg.epochs,fold=fold,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    cfg = cfg)
        # run.finish()
        # display(ipd.IFrame(run.url, width=1000, height=720))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser('Segmentation training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)