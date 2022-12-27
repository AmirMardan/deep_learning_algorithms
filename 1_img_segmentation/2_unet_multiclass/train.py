import torch
from torch.utils.data import DataLoader
from torch.optim import Adam 
from tqdm import tqdm
import tools
from data_scripts.dataset import get_loaders
from unet import UNet
from pathlib import Path
from torch import Tensor
from typing import Any, List, Optional, Union
import os 
import pandas as pd

def train(
    loader: DataLoader,
    model: UNet, 
    optimizer: torch.optim.Optimizer, 
    loss_fn,  
    scaler,
    device: torch.device,
    n_classes: int
):
    
    n_loader = len(loader)
    loop = tqdm(loader)
    loss_epoch = 0.0
    model.train()
    for idx, (img, target) in enumerate(loop):
        img = img.to(device=device)
        target = target.to(device=device)
         
        with torch.cuda.amp.autocast():
            prediction = model(img)
            if n_classes == 2:
                loss = loss_fn(prediction, target.unsqueeze(1))
            else:
                loss = loss_fn(prediction, target.to(dtype=torch.long))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        loss_epoch += loss.item()
    return loss_epoch/n_loader
        
        
def main(base_dir: str,
         img_height: Any, img_width: Any,
         batch_size: int, val_percentage: float,
         epochs: int, learning_rate: float,
         p: float, device: torch.device,
         path_to_save: str, save_frequency:int,
         loading_file: Optional[str]=None,
         features: List = [64, 128, 256, 512, 1024],
         n_classes: Optional[int] = 1,
         n_channels: int = 3, out_channels: int = 1) -> None:
    
    path_checkpoints = path_to_save + "/checkpoints"
    path_example = path_to_save + "/results/validation"
    Path(path_to_save+"/checkpoints/").mkdir(parents=True, exist_ok=True)
    Path(path_example).mkdir(parents=True, exist_ok=True)
    
    model = UNet(n_channels=n_channels,
                out_channels= out_channels, 
                features=features,
                bilinear=False).to(device=device)
    if loading_file:
        tools.load_checkpoint(model, path_to_save+"/checkpoints/"+loading_file)
        
    train_dl, val_dl = get_loaders(
        base_dir=base_dir,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size, 
        val_percentage=val_percentage,
        p=p)
    
    loss_fn = torch.nn.CrossEntropyLoss() if n_classes > 2 else torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    metrics = {
        "train_loss": [],
        "valid_loss": [],
        "dice_accuracy": [],
        "pixel_accuracy": [],
        "learning_rate": []
    }
    
    for epoch in range(1, epochs+1):
        loss = train(train_dl, model,
              optimizer=optimizer,
              loss_fn=loss_fn,
              scaler=scaler,
              device=device,
              n_classes=n_classes
            )
        metrics["train_loss"].append(loss)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # save model
        if (epoch % save_frequency == 0) or (epoch==epochs):
            tools.save_checkpoint(model, path_checkpoints+f"/checkpoint_{epoch}.tar")
        
        # check accuracy
        (pixel_accuracy,
         dice_accuracy,
         loss_val) = tools.check_accuracy(
            loader=val_dl, model=model,
            loss_fn=loss_fn,
            device=device,
            n_classes=n_classes
            )
        metrics["valid_loss"].append(loss_val)
        metrics["dice_accuracy"].append(dice_accuracy)
        metrics["pixel_accuracy"].append(pixel_accuracy)
        
        # print some exampels
        if epoch%2 == 0 or epoch==epochs:
            tools.save_predictions_as_image(
                val_dl, model=model,
                folder=path_example,
                device=device,
                n_classes=n_classes
            )
        print(f"Epoch {epoch}:",
              f"\ntrain loss: {metrics['train_loss'][-1]}",
              f"\nvalidation loss: {metrics['valid_loss'][-1]}"
              )

    df = pd.DataFrame(metrics)
    df.to_csv(path_checkpoints+"/metrics.csv", index=False)
    
    
if __name__ == "__main__":
    from default_values import *

    tools.seed_everything(seed=10)
    (base_path,
    path_to_save, num_epochs, 
    batch_size, val_percentage,
    learning_rate, device) = tools.arg_parser()
    
    main(base_path,
         IMAGE_HEIGHT, IMAGE_WIDTH,
         batch_size=batch_size,
         val_percentage=val_percentage,
         epochs=num_epochs, learning_rate=learning_rate,
         p=AUGMENTATION_P, device=device, 
         path_to_save=path_to_save, save_frequency=SAVE_FREQ,
         loading_file=LOAD_MODEL,
         features=FEATURES, n_classes=N_CLASSES,
         n_channels= N_CHANNELS, out_channels=OUT_CHANNELS)