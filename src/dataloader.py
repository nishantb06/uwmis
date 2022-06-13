import numpy as np
import pandas as pd
import torch 
import cv2

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import CFG
from utils.helper import load_img,load_msk

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

def get_mask_paths():
    """
    returns an updated df which has mask paths and other meta data added.
    this path might need to be changed to read the df
    """
    try:
        df = pd.read_csv('../input/uwmgi-mask-dataset/train.csv')
    except:
        print("change the path to read the csv file")
        return None
        
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len) # length of each rle mask
    df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy')

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
    df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len==0) # empty masks

    return df


def create_folds(df,cfg):
    skf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold

    return df

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = load_img(img_path)
        
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)

def get_transforms(train = True,cfg=None):
    data_transforms = {
        "train": A.Compose([
            A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=cfg.img_size[0]//20, max_width=cfg.img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        
        "valid": A.Compose([
            A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
    }
    if train==True:
        return data_transforms["train"] 
    else:
        return data_transforms['test']

#change where this function is used also

def prepare_loaders(fold,df,cfg, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32*5).query("empty==0")
        valid_df = valid_df.head(32*3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=get_transforms(train = True,cfg = cfg))
    valid_dataset = BuildDataset(valid_df, transforms=get_transforms(train = False,cfg = cfg))

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs if not debug else 20, 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_bs if not debug else 20, 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

train_loader, valid_loader = prepare_loaders(fold=0,df = create_folds(get_mask_paths()),debug=True)