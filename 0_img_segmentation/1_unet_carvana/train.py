from torch.optim import Adam 
from tqdm import tqdm
import tools
from data_scripts.dataset import get_loaders
from config import *
from unet import UNet
from pathlib import Path
import os 

def train(
    loader: DataLoader,
    model: UNet, 
    optimizer: torch.optim.Optimizer, 
    loss_fn,  
    scaler,
    device
):
    
    loop = tqdm(loader)
    loss_epoch = 0.0
    model.train()
    for idx, (img, target) in enumerate(loop):
        img = img.to(device=device)
        target = target.float().unsqueeze(1).to(device=device)
         
        with torch.cuda.amp.autocast():
            prediction = model(img)
            loss = loss_fn(prediction, target)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        loss_epoch += loss.item()
        
    return loss_epoch
        
        
def main(imgs_path: str, masks_path: str,
         img_height: Any, img_width: Any,
         batch_size: int, val_percentage: float,
         epochs: int, learning_rate: float,
         p: float, device: torch.device,
         path_to_save: str, save_frequency:int,
         loading_file: Optional[str]=None,
         features: List = [64, 128, 256, 512, 1024],
         n_classes: Optional[int] = 1, n_channels: int = 3) -> None:
    
    path_checkpoints = path_to_save + "/checkpoints"
    path_example = path_to_save + "/results"
    Path(path_to_save+"/checkpoints/").mkdir(parents=True, exist_ok=True)
    Path(path_example).mkdir(parents=True, exist_ok=True)
    
    model = UNet(n_channels=n_channels,
                n_classes= n_classes, 
                features=features,
                bilinear=False).to(device=device)
    if loading_file:
        tools.load_checkpoint(model, path_to_save+"/checkpoints/"+loading_file)
        
    train_dl, val_dl = get_loaders(
        imgs_path=imgs_path,
        masks_path=masks_path,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size, 
        val_percentage=val_percentage,
        p=p)
    # img, mask = next(iter(train_dl))
    # tools.show_batch(img, mask, batch_size)
    # plt.show()
    loss_fn = torch.nn.CrossEntropyLoss() if n_classes > 1 else torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    loss = []
    for epoch in range(1, epochs+1):
        loss.append(
            train(train_dl, model, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device
            )
        )

        # save model
        if (epoch % save_frequency == 0) or (epoch==epochs):
            tools.save_checkpoint(model, path_checkpoints+f"/checkpoint_{epoch}.tar")
        
        # check accuracy
        tools.check_accuracy(loader=val_dl, model=model,
                             device=device)
        
        # print some exampels
        if epoch%2 == 0 or epoch==epochs:
            tools.save_predictions_as_image(
                val_dl, model=model,
                folder=path_example,
                device=device
            )
        print(f"Epoch: {epoch} -> loss: {loss[-1]}")
    
    torch.save(loss, path_checkpoints + "/loss.pt")
    
    
if __name__ == "__main__":
    seed_everything(seed=10)
    data_dir, path_to_save = tools.arg_parser()

    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(__file__, "../../data/carvana"))
    imgs_path = data_dir + "/imgs/"
    masks_path = data_dir + "/masks/"
    
    if path_to_save is None:
        path_to_save = os.path.abspath(os.path.join(__file__, "../"))
    # print(path_to_save)
    
    main(imgs_path, masks_path,
         IMAGE_HEIGHT, IMAGE_WIDTH,
         batch_size=BATCH_SIZE,
         val_percentage=VAL_PERCENTAGE,
         epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
         p=AUGMENTATION_P, 
         path_to_save=path_to_save, save_frequency=SAVE_FREQ,
         loading_file=LOAD_MODEL,
         features= FEATURES, n_classes=N_CLASSES,
         n_channels= N_CHANNELS, device=DEVICE)