import utils.config as cfg
import numpy as np
import torch
from src.dataloader import train_loader,valid_loader

def main():
    imgs, msks = next(iter(train_loader))
    print(imgs.size(), msks.size())

if __name__=="__main__":
    main()