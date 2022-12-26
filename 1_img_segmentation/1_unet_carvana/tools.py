import torch
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
from unet import UNet
from torch import Tensor
import torchvision
import random, os
import numpy as np
from torch.utils.data import DataLoader
from typing import Any, List, Optional, Union
from default_values import *
from data_scripts.dataset import test_preparing

def show_test(
    img_path: str,
    model: UNet,
    img_height: Any,
    img_width: Any,
    device
    ):
    test = test_preparing(
        img_path=img_path,
        img_height=img_height,
        img_width=img_width
    )
    model = model.to(device=device)
    test =test.to(device=device)
    
    predicted = torch.sigmoid(model(test))
    predicted = (predicted>0.5).float()

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(121)
    ax.imshow(test.squeeze().permute(1, 2, 0).cpu())
    ax.set_title("True Image")
    ax.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(predicted.squeeze().detach().cpu(), cmap='gray')
    ax.set_title("Segmented Image")
    ax.axis("off")


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
    file:str,
    device: str) -> None:
    
    state = torch.load(f=file,
                       map_location=torch.device(device))
    model.load_state_dict(state)
    print("== Checkpoint is loaded! ==")


def check_accuracy(loader: DataLoader,
                   model: UNet,
                   device: torch.device,
                   loss_fn) -> Union[float,
                                     float, 
                                     float]:
                       
    n_loader = len(loader)
    
    num_corrects = 0
    num_pixels = 0
    dice_score = 0.0
    loss = 0.0
        
    model.eval()
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device=device) 
            mask = mask.to(device=device).unsqueeze(1)
            
            out = model(img)
            loss += loss_fn(out, mask)

            preds = torch.sigmoid(out)
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
    return ((num_corrects/num_pixels).item(), 
            (dice_score/n_loader).item(),
            (loss/n_loader).item())


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
    
    parser.add_argument(
        "--epochs", dest="epochs",
        help="Number of epochs for training",
        default=NUM_EPOCHS,
        type=int
    )
    
    parser.add_argument(
        "--batch_size", dest="batch_size",
        help="Size of batches",
        default=BATCH_SIZE,
        type=int
    )
    
    parser.add_argument(
        "--val_percentage", dest="val_percentage",
        help="Relative size of validation set in percentage [0, 1.0]",
        default=VAL_PERCENTAGE,
        type=float
    )
    
    parser.add_argument(
        "--learning_rate", dest="learning_rate",
        help="Learning rate",
        default=LEARNING_RATE,
        type=float
    )
    
    parser.add_argument(
        "--device", dest="device", 
        help="Name of device for training (cpu, cuda)",
        default=DEVICE,
        choices=["cpu", "cuda"]
    )
    
    args = parser.parse_args()
        
    return check_args(args)

def check_args(args: Namespace) -> Union[str,
                          str,
                          str]:
    data_dir = args.data_dir
    path_to_save = args.result_dir
    epochs = args.epochs
    batch_size = args.batch_size
    val_percentage = args.val_percentage
    learning_rate = args.learning_rate
    device = args.device
        
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(__file__, "../../data/carvana"))
    imgs_path = data_dir + "/imgs/"
    masks_path = data_dir + "/masks/"
    
    if not (os.path.isdir(imgs_path) and os.path.isdir(masks_path)):
        raise RuntimeError("data_dir must contain two folders called "
                           "imgs and masks for true images and masks, respectively.")
    
    if path_to_save is None:
        path_to_save = os.path.abspath(os.path.join(__file__, "../"))
    
    return (imgs_path, masks_path, path_to_save,
            epochs, batch_size, val_percentage,
            learning_rate, device)

