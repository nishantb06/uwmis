import segmentation_models_pytorch as smp
import torch
from utils.config import CFG


def build_model(cfg):
    model = smp.Unet(
        encoder_name=cfg.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=cfg.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(cfg.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model