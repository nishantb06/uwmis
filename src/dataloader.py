import numpy as np
import pandas as pd
import torch 

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


