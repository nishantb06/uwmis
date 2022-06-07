import numpy as np
import gc
import torch
import torch.optim as optim

from models.unet import build_model
from utils.config import CFG
from utils.loss_func import fetch_scheduler
from src.dataloader import prepare_loaders
from src.engine import run_training
from src.dataloader import create_folds,get_mask_paths


def main():

    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)

    for fold in range(1):
        print(f'#'*15)
        print(f'### Fold: {fold}')
        print(f'#'*15)
        # run = wandb.init(project='uw-maddison-gi-tract', 
        #                 config={k:v for k, v in dict(vars(CFG)).items() if '__' not in k},
        #                 anonymous=anonymous,
        #                 name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
        #                 group=CFG.comment,
        #                 )
        train_loader, valid_loader = prepare_loaders(fold=fold,
                                    df = create_folds(get_mask_paths()),
                                    debug=False)
        model     = build_model()
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = fetch_scheduler(optimizer)
        model, history = run_training(model, optimizer, scheduler,
                                    device=CFG.device,
                                    num_epochs=CFG.epochs,fold=fold,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader)
        # run.finish()
        # display(ipd.IFrame(run.url, width=1000, height=720))
    

if __name__=="__main__":
    main()