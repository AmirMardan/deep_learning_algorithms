import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from unet import UNet
from torch import Tensor
from config import *
import torchvision
import random, os
import numpy as np

def seed_everything(seed: int):
    
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def show_batch(img: Tensor,
               mask: Tensor,
               batch_size: int) -> None:
    
    fig, axs = plt.subplots(batch_size, 2, figsize=(4, 1*batch_size))
    count = 0
    for i, ax in enumerate(axs.flatten()):
        if i%2 == 0:
            ax.imshow(img[count, ...].detach().permute(1, 2, 0))
        else:
            ax.imshow(mask[count, ...].detach())
            count += 1
        ax.axis('off')
    if batch_size == 1:
        axs[0].set_title("Image")
        axs[1].set_title("Mask")
    else:
        axs[0, 0].set_title("Image")
        axs[0, 1].set_title("Mask")
        
        
def save_checkpoint(
    model: UNet,
    file:str) -> None:
    
    torch.save(model.state_dict(), file)
    print("== Checkpoint is saved! ==")
    

def load_checkpoint(
    model: UNet,
    file:str) -> None:
    
    state = torch.load(f=file)
    model.load_state_dict(state)
    print("== Checkpoint is loaded! ==")


def arg_parser():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_dir", dest="data_dir",
        help="Data directory"
    )
    parser.add_argument(
       "--result_dir", dest="result_dir",
       help="Directory to save checkpoints, loss, and some examples" 
    )
    
    args = parser.parse_args()
    data_dir = args.data_dir
    result_dir = args.result_dir
    
    return data_dir, result_dir

def check_accuracy(loader: DataLoader,
                   model: UNet,
                   device: torch.device) ->None:
    num_corrects = 0
    num_pixels = 0
    dice_score = 0.0
        
    model.eval()
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device=device) 
            mask = mask.to(device=device).unsqueeze(1)
            
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
            num_corrects += (preds == mask).sum()
            num_pixels += torch.numel(preds)
            # print(num_corrects/num_pixels)
            dice_score += (2 * (preds * mask).sum()) / (
                (preds + mask).sum() + 1e-8
            )
        
        print(
            f"Got {num_corrects}/{num_pixels} with accuracy {num_corrects/num_pixels*100:.2f}"
            f"\nDice score: {dice_score/len(loader)}"
        )


def save_predictions_as_image(
    loader: DataLoader,
    model: UNet,
    folder: str,
    device: torch.device
    ) -> None:
    
    model.eval()
    for i, (img, mask) in enumerate(loader):
        img = img.to(device=device)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(
            preds, fp=f"{folder}/pred_{i}.png"
        )
            
        torchvision.utils.save_image(
            mask.unsqueeze(1), fp=f"{folder}/true_{i}.png"
        )
    
